from typing import List

import asyncio
from io import BytesIO
from PIL import Image

import numpy as np
import torch
from torchvision import transforms  
from torchvision.ops import masks_to_boxes, box_iou

from starlette.requests import Request
from fastapi import FastAPI, Query, File

import ray
from ray import serve
from ray.serve.drivers import DAGDriver
from ray.serve.dag import InputNode 

from ptgctl import holoframe


app = FastAPI()


class Model:
    def load_image(self, data, format):
        if isinstance(data, list):
            return np.stack([self.load_image(d, format) for d in data], 0)

        if format == 'img':
            return np.array(Image.open(BytesIO(data)))
        return holoframe.load(data)['image']

    def __call__(self, im, format, *a, **kw):
        return self.forward(self.load_image(im, format), *a, **kw)


@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class DeticModel(Model):
    def __init__(self):
        from ptgprocess.detic import Detic
        assert torch.cuda.is_available()
        self.detic = Detic(one_class_per_proposal=False, conf_threshold=0.1).cuda()

    def forward(self, im):
        im = im[:,:,::-1]
        print(im.shape)
        outputs = self.detic(im)
        insts = outputs['instances'].to("cpu")

        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        box_confs = insts.box_scores.numpy()
        print(xyxy.shape, confs.shape)
        # combine (exact) duplicate bounding boxes
        xyxy_unique, ivs = self.detic.group_proposals(xyxy)
        xyxyn_unique = self.detic.boxnorm(xyxy_unique, *im.shape[:2])
        print(xyxyn_unique.shape)

        labels = self.detic.labels[class_ids]

        return xyxyn_unique, ivs, class_ids, labels, confs, box_confs


@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class EgoVLPModel(Model):
    def __init__(self):
        from ptgprocess.egovlp import EgoVLP
        assert torch.cuda.is_available()
        self.egovlp = EgoVLP().cuda()
        self.loaded_vocabs = {}

    def forward(self, im, recipe):
        print(im.shape)
        Z_images = self.egovlp.encode_video(torch.stack([
            self.egovlp.prepare_image(x) for x in im
        ], dim=1).cuda())
        pred = self.get_predictor(recipe)
        return pred(Z_images).detach().cpu().numpy()

    def get_predictor(self, name):
        if name in self.loaded_vocabs:
            return self.loaded_vocabs[name]
        self.loaded_vocabs[name] = pred = self.egovlp.get_predictor(name, '/home/bea/src/storage/fewshot')
        return pred

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
class EgoHosBoxModel(Model):
    def __init__(self, mode='obj1'):
        from ptgprocess.egohos import EgoHos
        self.hos = EgoHos(mode=mode)

    def forward(self, im):
        seg = self.hos(im)[0][:1]
        print(seg.shape, seg.sum(), seg.size)
        if not seg.any():
            return
        box = masks_to_boxes(torch.tensor(seg)).numpy()
        box = boxnorm(box, *im.shape[:2])
        print(box)
        return box

def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy




detic_model = None#DeticModel.bind()
egohos_model = None#EgoHosBoxModel.bind()
egovlp_model =EgoVLPModel.bind()

@serve.deployment
@serve.ingress(app)
class Server:
    def __init__(self, detic, egohos, egovlp):
        self.detic = detic
        self.egohos = egohos
        self.egovlp = egovlp

    
    @app.post('/detic')
    async def predict_detic(self, req: Request, format: str=Query('img')):
        data = await req.body()
        f = self.detic.remote(data, format)
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

    @app.post('/egohosbox')
    async def predict_egohos(self, req: Request, format: str=Query('img')):
        data = await req.body()
        f = self.egohos.remote(data, format)
        x = ray.get(await f)
        return x

    @app.post('/egovlp')
    async def predict(
        self, data: List[bytes] = File(),
        format: str=Query('img'), 
        recipe: str=Query('pinwheels', description='the recipe to predict'),

    ):
        f = self.egovlp.remote(data, format, recipe)
        x = ray.get(await f)
        return x

server = Server.bind(detic_model, egohos_model, egovlp_model)



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
