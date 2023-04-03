from typing import List

import asyncio
from io import BytesIO
from PIL import Image

import numpy as np
import torch
from torchvision import transforms  
from torchvision.ops import masks_to_boxes, box_iou

from starlette.requests import Request
from fastapi import FastAPI, Query, File, UploadFile

import ray

# ray.init()

from ray import serve
# from ray.serve.gradio_integrations import GradioIngress

# import gradio as gr
# from ray.serve.drivers import DAGDriver
# from ray.serve.dag import InputNode 

from ptgctl import holoframe
from ptgprocess.util import draw_boxes


app = FastAPI()


dualgpu = torch.cuda.device_count() == 2
detic_gpu = 0.5 if dualgpu else 0.25
omnivore_gpu = 0.5 if dualgpu else 0.25
egohos_gpu = 0.5 if dualgpu else 0.25
egovlp_gpu = 0.5 if dualgpu else 0.25
audio_sf_gpu = 0.5 if dualgpu else 0.25


class ImageModel:
    def load_image(self, data, format):
        if isinstance(data, list):
            return np.stack([self.load_image(d, format) for d in data], 0)

        if format == 'img':
            return np.array(Image.open(BytesIO(data)))
        return holoframe.load(data)['image']

    def __call__(self, im, format, *a, **kw):
        return self.forward(self.load_image(im, format), *a, **kw)

    def forward(self, im):
        raise NotImplemented


@serve.deployment(ray_actor_options={"num_gpus": detic_gpu})
class DeticModel(ImageModel):
    def __init__(self):
        from ptgprocess.detic import Detic
        assert torch.cuda.is_available()
        self.model = Detic(one_class_per_proposal=False, conf_threshold=0.3).cuda()
        self.model.eval()

    def register_vocab(self, name, vocab):
        pass

    def get_vocab(self):
        return self.model.labels

    def set_vocab(self, name):
        pass

    @torch.no_grad()
    def forward(self, im):
        im = im[:,:,::-1]
        outputs = self.model(im)
        xyxyn_unique, ivs, class_ids, labels, confs, box_confs = self.model.unpack_results(outputs, im)
        # return xyxyn_unique, ivs, class_ids, labels, confs, box_confs
        return {
            'xyxyn': xyxyn_unique, 
            'unique_index': ivs,
            'class_ids': class_ids, 
            'labels': labels, 
            'confidence': confs, 
            'box_confidence': box_confs,
        }




@serve.deployment(ray_actor_options={"num_gpus": egovlp_gpu})
class EgoVLPModel(ImageModel):
    def __init__(self):
        from ptgprocess.egovlp import EgoVLP, get_predictor
        assert torch.cuda.is_available()
        self.model = EgoVLP().cuda()
        self.model.eval()
        self._get_predictor = get_predictor
        self.loaded_vocabs = {}

    @torch.no_grad()
    def forward(self, im, recipe, as_dict=True):
        # encode video
        Z_images = self.model.encode_video(torch.stack([
            self.model.prepare_image(x) for x in im
        ], dim=1).cuda())
        # compare with vocab
        pred = self.get_predictor(recipe)
        sim = pred(Z_images).detach().cpu().numpy()
        if as_dict:
            return dict(zip(pred.vocab.tolist(), sim.tolist()))
        return sim

    def get_predictor(self, name):
        if name in self.loaded_vocabs:
            return self.loaded_vocabs[name]
        self.loaded_vocabs[name] = pred = self._get_predictor(
            self.model, name, '/home/bea/src/storage/fewshot')
        return pred



@serve.deployment(ray_actor_options={"num_gpus": omnivore_gpu})
class OmnivoreModel(ImageModel):
    def __init__(self):
        from ptgprocess.omnivore import Omnivore
        assert torch.cuda.is_available()
        self.model = Omnivore().cuda()
        self.model.eval()

    @torch.no_grad()
    def forward(self, ims, as_dict=True):
        # encode video
        ims = torch.stack([
            self.model.prepare_image(x) for x in ims
        ], dim=1).cuda()[None]
        
        actions = self.model(ims)
        verb, noun = self.model.soft_project_verb_noun(actions)

        if as_dict:
            return [
                [dict(zip(self.model.verb_labels, v.cpu().numpy())) for v in verb],
                [dict(zip(self.model.noun_labels, n.cpu().numpy())) for n in noun],
            ]
        return [verb, noun]


@serve.deployment(ray_actor_options={"num_gpus": audio_sf_gpu})
class AudioSlowFastModel:
    def __init__(self):
        from ptgprocess.audio_slowfast import AudioSlowFast
        assert torch.cuda.is_available()
        self.model = AudioSlowFast().cuda()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, y, sr, as_dict=True):
        # encode video
        specs = self.model.prepare_audio(np.frombuffer(y), sr)
        specs = [s.cuda() for s in specs]
        
        verb, noun = self.model(specs)

        if as_dict:
            return [
                [dict(zip(self.model.vocab[0], v.cpu().numpy())) for v in verb],
                [dict(zip(self.model.vocab[1], n.cpu().numpy())) for n in noun],
            ]
        return [verb, noun]


@serve.deployment(ray_actor_options={"num_gpus": 0})
class BBNYoloModel(ImageModel):
    def __init__(self):
        from ptgprocess.yolo import BBNYolo
        self.model = BBNYolo()
        # self.model.eval()

    @torch.no_grad()
    def forward(self, im):
        im = im[:,:,::-1] # numpy array rgb->bgr, Image rgb
        outputs = self.model(im)
        xyxyn, ivs, class_ids, labels, confs, box_confs = self.model.unpack_results(outputs)
        return {
            'xyxyn': xyxyn, 
            'class_ids': class_ids, 
            'labels': labels, 
            'confidence': confs, 
        }


@serve.deployment(ray_actor_options={"num_gpus": egohos_gpu})
class EgoHosBoxModel(ImageModel):
    def __init__(self, mode='obj1'):
        from ptgprocess.egohos import EgoHos
        self.hos = EgoHos(mode=mode)

    def forward(self, im):
        seg = self.hos(im)[0][:1]
        if not seg.any():
            return
        box = masks_to_boxes(torch.tensor(seg)).numpy()
        box = boxnorm(box, *im.shape[:2])
        return box

def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy


# @serve.deployment
# class MyGradioServer(GradioIngress):
#     def __init__(self, model):
#         self.model = model
#         super().__init__(lambda: gr.Interface(self.fanout, "image", "image"))
#         from fastapi import APIRouter
#         app = self.app
#         self.app = APIRouter()
#         self.app.include_router(app, prefix='/gio')

#     async def fanout(self, im):
#         result = ray.get(await self.model.remote(im))
#         return Image.fromarray(draw_boxes(im, result['xyxyn'], result['labels'])[:,:,::-1])





@serve.deployment
@serve.ingress(app)
class Server:
    def __init__(self, detic, egohos, egovlp, omnivore, audio_sf, bbn_yolo, gio_serve):
        self.detic = detic
        self.egohos = egohos
        self.egovlp = egovlp
        self.omnivore = omnivore
        self.audio_sf = audio_sf
        self.bbn_yolo = bbn_yolo
        self.gio_serve = gio_serve

    # @app.get('/gio')
    # async def serve_gio(self, req: Request):
    #     return ray.get(await self.gio_serve.remote(req))
    
    @app.post('/detic')
    async def predict_detic(self, req: Request, format: str=Query('img')):
        data = await req.body()
        f = self.detic.remote(data, format)
        x = ray.get(await f)
        return x
    
    @app.post('/bbn_yolo')
    async def predict_bbn_yolo(self, req: Request, format: str=Query('img')):
        data = await req.body()
        f = self.bbn_yolo.remote(data, format)
        x = ray.get(await f)
        return x
    
    @app.post('/omnivore')
    async def predict_omnivore(self, video: List[UploadFile], format: str=Query('img')):
        data = await asyncio.gather(*[f.read() for f in video])
        f = self.omnivore.remote(data, format)
        x = ray.get(await f)
        return x
    
    @app.post('/audio_slowfast')
    async def predict_audio_slowfast(self, audio: UploadFile, sr: int):
        data = await audio.read()
        f = self.audio_sf.remote(data, sr)
        x = ray.get(await f)
        return x
    
    # @app.post('/omnimix')
    # async def predict_omnimix(self, video: list[UploadFile], audio: UploadFile, format: str=Query('img')):
    #     audio_data, *data = await asyncio.gather(audio.read(), *[f.read() for f in video])
    #     f = self.omnivore.remote(data, audio_data, format)
    #     x = ray.get(await f)
    #     return x

    @app.post('/egovlp')
    async def predict(
        self, data: List[bytes] = File(),
        format: str=Query('img'), 
        recipe: str=Query('pinwheels', description='the recipe to predict'),
    ):
        f = self.egovlp.remote(data, format, recipe)
        x = ray.get(await f)
        return x
    
    @app.post('/egohosbox')
    async def predict_egohos(self, req: Request, format: str=Query('img')):
        data = await req.body()
        f = self.egohos.remote(data, format)
        x = ray.get(await f)
        return x

    @app.post('/detic_hoi')
    async def predict_detic_hoi(self, req: Request, format: str=Query('img')):
        data = await req.body()
        (
            (xyxyn_unique, ivs, class_ids, labels, confs, box_confs),
            hoi_box
        ) = ray.get(await asyncio.gather(
            self.detic.remote(data, format),
            self.egohos.remote(data, format)
        ))

        if hoi_box is not None:
            ious = box_iou(torch.as_tensor(hoi_box), torch.as_tensor(xyxyn_unique))[0].numpy()
        else:
            ious = np.zeros(len(xyxyn_unique))
        return xyxyn_unique, ivs, ious, class_ids, labels, confs, box_confs

    

    





# def run():
detic_model = None#DeticModel.bind()
egohos_model = None#EgoHosBoxModel.bind()
egovlp_model = EgoVLPModel.bind()
omnivore_model = OmnivoreModel.bind()
bbn_yolo_model = BBNYoloModel.bind()
audio_sf_model = AudioSlowFastModel.bind()
gio_serve = None#MyGradioServer.bind(bbn_yolo_model)
# server = Server.bind(detic_model, egohos_model, egovlp_model)
server = Server.bind(detic_model, egohos_model, egovlp_model, omnivore_model, audio_sf_model, bbn_yolo_model, gio_serve)
    # serve.run(server)

# from ray.serve.gradio_integrations import GradioIngress
# import gradio as gr

# @serve.deployment
# class DeticGradioServer(GradioIngress):
#     def __init__(self, detic_model):
#         self.model = detic_model
#         super().__init__(lambda: gr.Interface(self.fanout, gr.Video(), "playable_video"))

#     async def fanout(self, video):
#         refs = await asyncio.gather(self._d1.remote(text), self._d2.remote(text))
#         [result1, result2] = ray.get(refs)
#         return (
#             f"[Generated text version 1]\n{result1}\n\n"
#             f"[Generated text version 2]\n{result2}"
#         )


# model1 = DeticModel.bind()
# model2 = EgoHosModel.bind(limit=1)
# 
# with InputNode() as user_input:
#     output1 = model1.forward.bind(user_input)
#     output2 = model2.forward.bind(user_input)
#     combine_output = combine.bind([output1, output2])
# 
# 
# async def holo_resolver(request: Request):
#     return holoframe.load(await request.json())
# 
# 
# graph = DAGDriver.bind(combine_output, http_adapter=holo_resolver)
# serve.run(graph)
# 
# #image_model = ObjectModel.bind()
# #serve.run(image_model)
