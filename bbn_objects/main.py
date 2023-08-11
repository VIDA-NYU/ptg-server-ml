
import os
import orjson
import asyncio
from collections import OrderedDict, defaultdict, deque

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import torch
from torchvision.ops import masks_to_boxes, box_iou

import ptgctl
from ptgctl import holoframe
import ptgctl.util

from ptgprocess.detic import Detic
from ptgprocess.yolo import BBNYolo
from ptgprocess.egohos import EgoHos

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


RECORDING_RAW_DIR = '/src/app/recordings/raw'
RECORDING_POST_DIR = '/src/app/recordings/post'

DEFAULT_VOCAB = 'lvis'

class BBNObjectSession:
    low_conf = 0.1
    mid_conf = 0.3
    hand_decay = 0.5
    mean_person_confs_decay = 0
    def __init__(self, model, egohos, prefix=None):
        self.model = model
        self.egohos = egohos
        self.out_sids = [
            f'{prefix or ""}detic:image',
            f'{prefix or ""}detic:image:v2',
            f'{prefix or ""}detic:image:for3d',
            f'{prefix or ""}detic:hands',
        ]

    def on_image(self, image, **params):
        im_params = self.pack_image_params(image, **params)
        outputs = self.model(image[:,:,::-1])
        xyxyn, _, class_ids, labels, confs, _ = self.model.unpack_results(outputs)
        xyxy = outputs[0].to("cpu").boxes.xyxy.numpy()
        ious = self.get_egohos_overlap(image, xyxy)
        objsv1 = self.as_v1_objs(xyxyn, confs, class_ids)
        objsv2 = self.as_v2_objs(outputs[0].boxes.xyxy, confs, class_ids, labels, ious)
        hand = self.get_hand_presence(self.model.labels[class_ids], confs)
        return dict(zip(self.out_sids, [
            objsv1,
            objsv2,
            { 'objects': objsv1, 'image_params': im_params },
            hand,
        ]))
    
    def pack_image_params(self, image, **main):
        return {
            'shape': image.shape,
            'focal': [main['focalX'], main['focalY']],
            'principal': [main['principalX'], main['principalY']],
            'cam2world': main['cam2world'].tolist(),
        }
    
    def as_v1_objs(self, xyxy, confs, class_ids):
        # filter low confidence
        ivs = confs > self.mid_conf
        objects = []
        for xy, c, cid in zip(xyxy[ivs], confs[ivs], class_ids[ivs]):
            objects.append({
                "xyxyn": xy.tolist(),
                "confidence": c,
                "class_id": cid,
                "label": self.model.labels[cid],
            })
        return objects
    
    def as_v2_objs(self, xyxy, confs, class_ids, labels, ious):
        objects = []
        for xy, c, cid, l, iou in zip(xyxy, confs, class_ids, labels, ious):
            # make object
            objects.append({
                "xyxyn": xy.tolist(),
                "class_ids": [cid],
                "labels": [l],
                "confidences": [c],
                # "box_confidences": [boxc],
                "hoi_iou": iou,
            })
        return objects


    def get_hand_presence(self, labels, confs):
        # use person detection as a reasoning signal
        is_person = labels == 'person'
        person_confs = confs[is_person] if is_person.sum() else np.zeros((1,))
        mean_person_confs = np.nanmean(person_confs)
        self.mean_person_confs_decay = (
            (1 - self.hand_decay) * mean_person_confs + 
            self.hand_decay * self.mean_person_confs_decay)
        return {
            'n_person_boxes': is_person.sum(),
            'max_conf': np.max(person_confs, initial=0),
            'mean_conf': mean_person_confs,
            'mean_conf_smooth': self.mean_person_confs_decay,
        }
    
    def get_egohos_overlap(self, im, xyxyn):
        # # get hand object bboxes and get overlap with bboxes
        segs = self.egohos(im)[0]
        obj1_seg = torch.as_tensor(segs[:1])
        if obj1_seg.any():
            seg_box = masks_to_boxes(obj1_seg)
            ious = box_iou(seg_box, torch.as_tensor(xyxyn))[0].numpy()
        else:
            ious = np.zeros(len(xyxyn))
        return ious







class DeticObjectSession:
    low_conf = 0.1
    mid_conf = 0.3

    hand_decay = 0.5
    mean_person_confs_decay = 0

    def __init__(self, model, egohos, labels, display_labels=None, valid_labels=None, prefix=None):
        self.model = model
        self.egohos = egohos
        self.labels = labels
        self.display_labels = labels if display_labels is None else display_labels
        self.valid_labels = valid_labels
        self.out_sids = [
            f'{prefix or ""}detic:image',
            f'{prefix or ""}detic:image:v2',
            f'{prefix or ""}detic:image:for3d',
            f'{prefix or ""}detic:hands',
        ]

    def on_image(self, **data):
        im, im_params = self.load_image(**data)
        (xyxy, ivs, confs, class_ids, box_confs, ious) = self.predict(im)

        objsv1 = self.as_v1_objs(xyxy, ivs, confs, class_ids)
        objsv2 = self.as_v2_objs(xyxy, ivs, confs, class_ids, box_confs, ious)
        hand = self.get_hand_presence(self.labels[class_ids], confs)
        return dict(zip(self.out_sids, [
            objsv1,
            objsv2,
            { 'objects': objsv1, 'image_params': im_params },
            hand,
        ]))

    def load_image(self, **main):
        im = main['image'][:,:,::-1]
        im_params = {
            'shape': im.shape,
            'focal': [main['focalX'], main['focalY']],
            'principal': [main['principalX'], main['principalY']],
            'cam2world': main['cam2world'].tolist(),
        }
        return im, im_params
    
    def predict(self, im):
        # predict objects
        outputs = self.model(im)
        insts = outputs["instances"].to("cpu")

        # ignore counter-factual labels e.g. feet
        if self.valid_labels is not None:
            insts = insts[np.where(self.valid_labels[insts.pred_classes])[0]]

        # get object info
        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        box_confs = insts.box_scores.numpy()

        # combine (exact) duplicate bounding boxes
        xyxy_unique, ivs = self.model.group_proposals(xyxy)
        # convert image boundaries to [0, 1]
        xyxyn_unique = boxnorm(xyxy_unique.copy(), *im.shape[:2])

        # # get hand object bboxes and get overlap with bboxes
        segs = self.egohos(im)[0]
        obj1_seg = torch.as_tensor(segs[:1])
        if obj1_seg.any():
            seg_box = masks_to_boxes(obj1_seg)
            ious = box_iou(seg_box, torch.as_tensor(xyxy_unique))[0].numpy()
        else:
            ious = np.zeros(len(xyxy_unique))
        #ious = np.zeros(len(xyxy_unique))

        return xyxyn_unique, ivs, confs, class_ids, box_confs, ious
    
    def as_v1_objs(self, xyxy, ivs, confs, class_ids):
        # filter low confidence
        ivs = ivs & (confs[None] > self.mid_conf)
        has_mid_conf = ivs.any(axis=1)

        objects = []
        for xy, iv in zip(xyxy[has_mid_conf], ivs[has_mid_conf]):
            # select relevant
            c_ids = class_ids[iv]
            c = confs[iv]
            # get max confidence prediction
            i = np.argmax(c)
            # make object
            objects.append({
                "xyxyn": xy.tolist(),
                "confidence": c[i],
                "class_id": c_ids[i],
                "label": self.display_labels[c_ids[i]],
            })
        return objects
    
    def as_v2_objs(self, xyxy, ivs, confs, class_ids, box_confs, ious):
        objects = []
        for xy, iv, iou in zip(xyxy, ivs, ious):
            # select relevant
            c_ids = class_ids[iv]
            c = confs[iv]
            boxc = box_confs[iv]
            # sort by confidence
            ix = np.argsort(c)[::-1]
            c_ids = c_ids[ix]
            c = c[ix]
            boxc = boxc[ix]
            # make object
            objects.append({
                "xyxyn": xy.tolist(),
                "class_ids": c_ids.tolist(),
                "labels": self.display_labels[c_ids].tolist(),
                "confidences": c.tolist(),
                "box_confidences": boxc.tolist(),
                "hoi_iou": iou,
            })
        return objects

    def get_hand_presence(self, labels, confs):
        # use person detection as a reasoning signal
        is_person = labels == 'person'
        person_confs = confs[is_person] if is_person.sum() else np.zeros((1,))
        mean_person_confs = np.nanmean(person_confs)
        self.mean_person_confs_decay = (
            (1 - self.hand_decay) * mean_person_confs + 
            self.hand_decay * self.mean_person_confs_decay)
        return {
            'n_person_boxes': is_person.sum(),
            'max_conf': np.max(person_confs, initial=0),
            'mean_conf': mean_person_confs,
            'mean_conf_smooth': self.mean_person_confs_decay,
        }

def get_vocab_lists_from_recipe_json(recipe, vocab_keys, extra_vocab, ignore_vocab):
    # get mapping between detic vocab and recipe vocab
    vocab_map = {}
    vocab = list(extra_vocab) + list(ignore_vocab)
    for k in vocab_keys:
        v = recipe.get(k) or []
        if isinstance(v, list):
            vocab.extend(v)
        elif isinstance(v, dict):
            vocab.extend(x for xs in v.values() for x in xs)
            vocab_map.update({x: k for k, xs in v.items() for x in xs})

    # set vocab
    labels = np.array(vocab)
    display_labels = np.array([vocab_map.get(x,x) for x in vocab])
    valid_labels = np.array([x not in ignore_vocab or x in vocab_map for x in vocab])
    return labels, display_labels, valid_labels



def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy









class RecipeExit(Exception):
    pass


n_gpus = torch.cuda.device_count()

class ObjectApp:
    low_conf = 0.1
    vocab_keys = ['ingredient_objects', 'tool_objects']
    EXTRA_VOCAB = ['person']
    IGNORE_VOCAB = ['feet']

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'bbn_yolo',
                              password=os.getenv('API_PASS') or 'bbn_yolo')
        self.bbn_model = BBNYolo()
        self.detic_model = Detic(device='cuda:0', one_class_per_proposal=False, conf_threshold=self.low_conf)
        self.egohos = EgoHos(device=f'cuda:{1 if n_gpus > 1 else 0}')
        # self.bbn_model.eval()
        self.detic_model.eval()
        self.egohos.eval()

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling and recipe status watching.'''
        while True:
            try:
                recipe_id = self.api.session.current_recipe()
                while not recipe_id: # or recipe_id == 'tourniquet':
                    print("waiting for recipe to be activated")
                    recipe_id = await self._watch_recipe_id(recipe_id)
                
                print("Starting recipe:", recipe_id)
                await self.run_recipe(recipe_id, *a, **kw)
            except RecipeExit:
                pass
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id=None):
        recipe_id = recipe_id or b''
        async with self.api.data_pull_connect('event:recipe:id') as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    def start_session(self, recipe_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if recipe_id == 'tourniquet':
            print("BBN yolo model")
            self.session = BBNObjectSession(self.bbn_model, self.egohos, prefix=prefix)
        else:
            if recipe_id:
                recipe = self.api.recipes.get(recipe_id)
                vocab, display_labels, valid_labels = get_vocab_lists_from_recipe_json(
                    recipe, self.vocab_keys, self.EXTRA_VOCAB, self.IGNORE_VOCAB)
                print("Detic model:", vocab)
                self.detic_model.set_vocab(vocab)
            else:
                print('no recipe. using default vocab:', DEFAULT_VOCAB)
                vocab = self.detic_model.set_vocab(DEFAULT_VOCAB)
                display_labels = valid_labels = None
            self.session = DeticObjectSession(
                self.detic_model, self.egohos, 
                vocab, display_labels, valid_labels, 
                prefix=prefix)

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id)

        # stream ids
        in_sid = f'{prefix or ""}main'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(self.session.out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            if d.decode() != recipe_id:
                                print("recipe changed", recipe_id, '->', d, flush=True)
                                return 

                        # predict actions
                        preds = self.session.on_image(**holoframe.load(d))
                        if preds is not None:
                            preds = {k: v for k, v in preds.items() if v is not None}
                            await ws_push.send_data(
                                [jsondump(x) for x in preds.values()], 
                                list(preds.keys()), 
                                [t]*len(preds))

    @torch.no_grad()
    def run_offline(self, recording_id, recipe_id, prefix=None):
        from contextlib import ExitStack
        from ptgprocess.util import VideoInput
        from ptgprocess.record import RawReader, RawWriter, JsonWriter
        self.start_session(recipe_id, prefix=prefix)

        with ExitStack() as exits:
            raw_writer = None
            if os.path.isfile(recording_id):
                loader = lambda x: {'image': x, 'focalX': 0, 'focalY': 0, 'principalX': 0, 'principalY': 0, 'cam2world': np.zeros((4,4))}
                reader = VideoInput(recording_id)
                post_dir = os.path.splitext(recording_id)[0]
            else:
                loader = lambda d: holoframe.load(d)
                raw_dir = os.path.join(RECORDING_RAW_DIR, recording_id)
                post_dir = os.path.join(RECORDING_POST_DIR, recording_id)
                print("raw directory:", raw_dir)
                reader = RawReader(os.path.join(raw_dir, 'main'))
                raw_writer = RawWriter(self.session.out_sids, raw_dir)
                exits.enter_context(raw_writer)
            os.makedirs(post_dir, exist_ok=True)
            json_writers = {
                s: JsonWriter(s, post_dir)
                for s in self.session.out_sids
            }
            for w in json_writers.values():
                exits.enter_context(w)
            exits.enter_context(reader)

            for ts, d in reader:
                if isinstance(ts, float):
                    ts = ptgctl.util.format_epoch_time(ts)
                out = self.session.on_image(**loader(d))
                if raw_writer:
                    raw_writer.write(jsondump(out), ts)
                for k, x in out.items():
                    json_writers[k].write(jsondump({'data': x, 'timestamp': ts}), ts)




def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(ObjectApp)
