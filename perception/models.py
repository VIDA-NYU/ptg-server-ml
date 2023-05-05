from typing import List
from functools import lru_cache
import asyncio
from PIL import Image

import numpy as np
import torch
import ray

# ray.init()



# dualgpu = torch.cuda.device_count() == 2
# detic_gpu = 0.5 if dualgpu else 0.25
# omnivore_gpu = 0.5 if dualgpu else 0.25
# egohos_gpu = 0.5 if dualgpu else 0.25
# egovlp_gpu = 0.5 if dualgpu else 0.25
# audio_sf_gpu = 0.5 if dualgpu else 0.25


@lru_cache(maxsize=1)
def get_omnivore():
    from ptgprocess.omnivore import Omnivore
    # assert torch.cuda.is_available()
    omni_model = Omnivore().cuda()
    omni_model.eval()
    return omni_model

@lru_cache(maxsize=1)
def get_yolo(skill):
    from ptgprocess.yolo import BBNYolo
    yolo_model = BBNYolo(skill=skill)
    return yolo_model

@lru_cache(maxsize=1)
def get_clip(device):
    import clip
    return clip.load("ViT-B/16", device=device, jit=False)

@lru_cache(maxsize=1)
def get_omnigru():
    from ptgprocess.omnimix import OmniGRU2
    mix_model = OmniGRU2().cuda()
    mix_model.eval()
    return mix_model

@ray.remote(name='omnimix_full', num_gpus=1)
class AllInOneModel:
    def __init__(self, skill='M2'):
        self.device = 'cuda'#torch.cuda.current_device()
        self.omni_model = get_omnivore()
        self.clip_model, self.clip_transform = get_clip(self.device)
        self.mix_model = get_omnigru()
        skill and self.load_skill(skill)

    def load_skill(self, skill):
        self.yolo_model = get_yolo(skill=skill)
        # self.hidden = None

    @torch.no_grad()
    def forward(self, ims_bgr, hidden):
        ims_rgb = [im_bgr[:,:,::-1] for im_bgr in ims_bgr]
        im_bgr = ims_bgr[-1]
        im_rgb = ims_rgb[-1]

        # get action embeddings
        X = torch.stack([
            self.omni_model.prepare_image(x) for x in ims_bgr
        ], dim=1).cuda()[None]
        z_omni = self.omni_model.model(X, input_type="video")

        # get bounding boxes
        outputs = self.yolo_model(im_bgr)
        boxes = outputs[0].boxes

        # get clip patches
        X = torch.stack([
            self.clip_transform(Image.fromarray(x)).to(self.device)
            for x in [im_bgr] + extract_patches(im_bgr, boxes.xywh.cpu().numpy(), (224,224))
        ])
        z_clip = self.clip_model.encode_image(X)

        # concatenate with boxes
        z_clip_frame = torch.cat([z_clip[:1], torch.tensor([[0, 0, 1, 1, 1]]).to(self.device)], axis=1)
        z_clip_patch = torch.cat([z_clip[1:], boxes.xywhn, boxes.conf[:, None]], axis=1)
        # pad boxes to size
        z_clip_patch_pad = torch.zeros(
            (max(self.MAX_OBJECTS - z_clip_patch.shape[0], 0), 
             z_clip_patch.shape[1])).to(self.device)
        z_clip_patch = torch.cat([z_clip_patch, z_clip_patch_pad])[:self.MAX_OBJECTS]

        # get mixin
        x = z_omni[None].float(), z_clip_frame[None].float(), z_clip_patch[None,None].float()
        steps, hidden = self.mix_model(x, hidden)
        steps = torch.softmax(steps[0,0,:-2], dim=-1)

        # prepare objects
        boxes = boxes.cpu()
        objects = as_v1_objs(
            boxes.xyxyn.numpy(), 
            boxes.conf.numpy(), 
            boxes.cls.numpy(), 
            self.yolo_model.labels[boxes.cls.int().numpy()], 
            conf_threshold=0.5)
        return steps.cpu(), objects, hidden
    
    def forward_boxes(self, im):
        im_rgb = im[:,:,::-1]
        # get bounding boxes
        outputs = self.yolo_model(im_rgb)
        boxes = outputs[0].boxes
        # prepare objects
        boxes = boxes.cpu()
        objects = as_v1_objs(
            boxes.xyxyn.numpy(), boxes.conf.numpy(), 
            boxes.cls.numpy(), self.yolo_model.labels[boxes.cls.int().numpy()], 
            conf_threshold=0.5)
        return objects

    MAX_OBJECTS = 25



# @ray.remote(name='detic', num_gpus=0.1)
# class DeticModel:
#     def __init__(self):
#         from ptgprocess.detic import Detic
#         # assert torch.cuda.is_available()
#         self.model = Detic(one_class_per_proposal=False, conf_threshold=0.3).cuda()
#         self.model.eval()

#     def register_vocab(self, name, vocab):
#         pass

#     def get_vocab(self):
#         return self.model.labels

#     def set_vocab(self, name):
#         pass

#     def forward(self, im):
#         im = im[:,:,::-1]
#         outputs = self.model(im)
#         xyxyn_unique, ivs, class_ids, labels, confs, box_confs = self.model.unpack_results(outputs, im)
#         # return xyxyn_unique, ivs, class_ids, labels, confs, box_confs
#         return {
#             'xyxyn': xyxyn_unique, 
#             'unique_index': ivs,
#             'class_ids': class_ids, 
#             'labels': labels, 
#             'confidence': confs, 
#             'box_confidence': box_confs,
#         }




# @ray.remote(name='egovlp', num_gpus=0.1)
# class EgoVLPModel:
#     def __init__(self):
#         from ptgprocess.egovlp import EgoVLP, get_predictor
#         assert torch.cuda.is_available()
#         self.model = EgoVLP().cuda()
#         self.model.eval()
#         self._get_predictor = get_predictor
#         self.loaded_vocabs = {}

#     def forward(self, im, recipe, as_dict=True):
#         # encode video
#         Z_images = self.model.encode_video(torch.stack([
#             self.model.prepare_image(x) for x in im
#         ], dim=1).cuda())
#         # compare with vocab
#         pred = self.get_predictor(recipe)
#         sim = pred(Z_images).detach().cpu().numpy()
#         if as_dict:
#             return dict(zip(pred.vocab.tolist(), sim.tolist()))
#         return sim

#     def get_predictor(self, name):
#         if name in self.loaded_vocabs:
#             return self.loaded_vocabs[name]
#         self.loaded_vocabs[name] = pred = self._get_predictor(
#             self.model, name, '/home/bea/src/storage/fewshot')
#         return pred



# @ray.remote(name='omnivore', num_gpus=0.1)
# class OmnivoreModel:
#     def __init__(self):
#         from ptgprocess.omnivore import Omnivore
#         # assert torch.cuda.is_available()
#         self.model = Omnivore().cuda()
#         self.model.eval()

#     def encode_images(self, ims):
#         # encode video
#         ims = torch.stack([
#             self.model.prepare_image(x) for x in ims
#         ], dim=1).cuda()[None]
        
#         z = self.model.model(ims, input_type="video")
#         return z

#     def forward(self, ims, as_dict=True):
#         # encode video
#         ims = torch.stack([
#             self.model.prepare_image(x) for x in ims
#         ], dim=1).cuda()[None]
        
#         actions = self.model(ims)
#         verb, noun = self.model.soft_project_verb_noun(actions)

#         if as_dict:
#             return [
#                 [dict(zip(self.model.verb_labels, v.cpu().numpy())) for v in verb],
#                 [dict(zip(self.model.noun_labels, n.cpu().numpy())) for n in noun],
#             ]
#         return [verb, noun]


# @ray.remote(name='audio_slowfast', num_gpus=0.1)
# class AudioSlowFastModel:
#     def __init__(self):
#         from ptgprocess.audio_slowfast import AudioSlowFast
#         assert torch.cuda.is_available()
#         self.model = AudioSlowFast().cuda()
#         self.model.eval()

#     @torch.no_grad()
#     def __call__(self, y, sr, as_dict=True):
#         # encode video
#         specs = self.model.prepare_audio(np.frombuffer(y), sr)
#         specs = [s.cuda() for s in specs]
        
#         verb, noun = self.model(specs)

#         if as_dict:
#             return [
#                 [dict(zip(self.model.vocab[0], v.cpu().numpy())) for v in verb],
#                 [dict(zip(self.model.vocab[1], n.cpu().numpy())) for n in noun],
#             ]
#         return [verb, noun]


# @ray.remote(name='bbn_yolo', num_gpus=0.1)
# class BBNYoloModel:
#     def __init__(self):
#         from ptgprocess.yolo import BBNYolo
#         self.model = BBNYolo(skill='tourniquet')
#         # self.model.eval()

#     def forward(self, im):
#         im = im[:,:,::-1] # numpy array rgb->bgr, Image rgb
#         outputs = self.model(im)
#         xywhn, ivs, class_ids, labels, confs, box_confs = self.model.unpack_results(outputs, 'xywhn')
#         return {
#             'xywhn': xywhn, 
#             'class_ids': class_ids, 
#             'labels': labels, 
#             'confidence': confs, 
#         }



# @ray.remote(name='clip_patches', num_gpus=0.1)
# class ClipPatchModel:
#     def __init__(self):
#         import clip
#         self.device = 'cpu'#torch.cuda.current_device()
#         self.model, self.transform = clip.load("ViT-B/16", device=self.device, jit=False)

#     def forward(self, im, boxes):
#         # X = self.transform(Image.fromarray(im))[None].to(self.device)
#         X = torch.stack([
#             self.transform(Image.fromarray(x)).to(self.device)
#             for x in [im] + extract_patches(im, boxes, (224,224))
#         ])
#         return self.model.encode_image(X)


def extract_image_patch(image, xywh, patch_shape=None):
    bbox = np.asarray(xywh)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width
    w, h = wh = np.asarray(image.shape[:2][::-1])

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    # bbox[0] *= w
    # bbox[1] *= h
    # bbox[2] *= w
    # bbox[3] *= h
    bbox = bbox.astype(int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(wh - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None

    # 
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


def extract_patches(image, boxes, patch_shape=None):
    patches = []
    for box in boxes:
        patch = extract_image_patch(image, box, patch_shape=patch_shape)
        if patch is None:
            print(f"WARNING: Failed to extract image patch: {box}.")
            patch = np.random.uniform(0, 255, (*patch_shape, 3) if patch_shape else image.shape, dtype=np.uint8)
        patches.append(patch)
    return patches


def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy


# @ray.remote(name="omnimix", num_gpus=0.1)
# class OmnimixModel:
#     def __init__(self):
#         from ptgprocess.omnimix import OmniGRU2
#         self.model = OmniGRU2()
#         # self.model.eval()

#     def forward(self, x, hidden):
#         z_omni, z_clip, z_patches = x
#         steps, hidden = self.model(x, hidden)
#         return {
#             'steps': steps, 
#             'hidden': hidden, 
#         }

def as_v1_objs(xyxy, confs, class_ids, labels, conf_threshold=0.5):
        # filter low confidence
        objects = []
        for xy, c, cid, l in zip(xyxy, confs, class_ids, labels):
            if c < conf_threshold: continue
            objects.append({
                "xyxyn": xy.tolist(),
                "confidence": c,
                "class_id": cid,
                "label": l,
            })
        return objects