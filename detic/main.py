import os
import orjson
import asyncio
import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import torch
import numpy as np
from torchvision.ops import masks_to_boxes, box_iou

import ptgctl
from ptgctl import holoframe
import ptgctl.util
from ptgprocess.detic import Detic
from ptgprocess.egohos import EgoHos

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')


RECORDING_RAW_DIR = '/src/app/recordings/raw'
RECORDING_POST_DIR = '/src/app/recordings/post'
#import ray


# @ray.remote
# class DeticModel:
#     def __init__(self, conf):
#         self.model = Detic(one_class_per_proposal=False, conf_threshold=conf)
# 
#     def predict(self, im):
#         # predict objects
#         outputs = self.model(im)
#         insts = outputs["instances"].to("cpu")
# 
#         # get object info
#         xyxy = insts.pred_boxes.tensor.numpy()
#         class_ids = insts.pred_classes.numpy().astype(int)
#         confs = insts.scores.numpy()
#         box_confs = insts.box_scores.numpy()
# 
#         return xyxy, class_ids, confs, box_confs
# 
#     def set_vocab(self, vocab):
#         self.model.set_vocab(vocab)
#         return np.array(self.model.labels)
# 
# 
# @ray.remote
# class EgoHosModel:
#     def __init__(self):
#         self.model = EgoHos()
# 
#     def predict_box(self, im):
#         segs = self.model(im)[0]
#         obj1_seg = torch.as_tensor(segs[:1])
#         return masks_to_boxes(obj1_seg) if obj1_seg.any() else None
# 

DEFAULT_VOCAB = 'lvis'

class DeticApp:
    vocab_keys = ['ingredient_objects', 'tool_objects']

    EXTRA_VOCAB = ['person']
    IGNORE_VOCAB = ['feet']

    low_conf = 0.1
    mid_conf = 0.3

    hand_decay = 0.5

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'detic',
                              password=os.getenv('API_PASS') or 'detic')
        self.model = Detic(device='cuda:1', one_class_per_proposal=False, conf_threshold=self.low_conf)
        self.egohos = EgoHos(device='cuda:0')

        #self.detic_model = DeticModel.remote(self.low_conf)
        #self.egohos_model = EgoHosModel.remote()

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                await self._run(*a, **kw)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)
    
    async def _run(self):
        # stream ids
        in_sids = ['main']
        out_sids = ['detic:image', 'detic:image:v2', 'detic:image:for3d', 'detic:hands']
        recipe_sid = 'event:recipe:id'
        vocab_sid = 'event:recipes'

        self.change_recipe(self.api.session.current_recipe())

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect(in_sids + [recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, buffer in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        tms = int(t.split('-')[0])
                        
                        # watch for recipe changes
                        if sid in {recipe_sid, vocab_sid}:
                            #await writer.write(b'[]', b'[]')
                            await ws_push.send_data([b'[]']*len(out_sids), out_sids, [t]*len(out_sids))
                            log.debug(buffer.decode('utf-8'))
                            self.change_recipe(buffer.decode('utf-8'))
                            continue
                        
                        # compute box
                        im, im_params = self.load_image(buffer)
                        (xyxy, ivs, confs, class_ids, box_confs, ious) = self.predict(im)

                        objsv1 = self.as_v1_objs(xyxy, ivs, confs, class_ids)
                        objsv2 = self.as_v2_objs(xyxy, ivs, confs, class_ids, box_confs, ious)
                        hand = self.get_hand_presence(self.labels[class_ids], confs)

                        await ws_push.send_data([
                            jsondump(objsv1),
                            jsondump(objsv2),
                            jsondump({ 'objects': objsv1, 'image_params': im_params }),
                            jsondump(hand),
                        ], out_sids, [t]*len(out_sids))

    def load_image(self, buffer):
        main = holoframe.load(buffer)
        im = main['image'][:,:,::-1]
        im_params = {
            'shape': im.shape,
            'focal': [main['focalX'], main['focalY']],
            'principal': [main['principalX'], main['principalY']],
            'cam2world': main['cam2world'].tolist(),
        }
        return im, im_params
                        
    def change_recipe(self, recipe_id):
        self.mean_person_confs_decay = 0
        if not recipe_id:
            print('no recipe. using default vocab:', DEFAULT_VOCAB)
            #vocab = ray.get(self.detic_model.set_vocab.remote(DEFAULT_VOCAB))
            self.model.set_vocab(DEFAULT_VOCAB)
            vocab = self.model.labels
            self.labels = self.display_labels = vocab
            self.valid_labels = None
            return
        print('using recipe', recipe_id)
        recipe = self.api.recipes.get(recipe_id)

        # get mapping between detic vocab and recipe vocab
        vocab_map = {}
        vocab = list(self.EXTRA_VOCAB) + list(self.IGNORE_VOCAB)
        for k in self.vocab_keys:
            v = recipe.get(k) or []
            if isinstance(v, list):
                vocab.extend(v)
            elif isinstance(v, dict):
                vocab.extend(x for xs in v.values() for x in xs)
                vocab_map.update({x: k for k, xs in v.items() for x in xs})

        # set vocab
        self.model.set_vocab(vocab)
        #self.detic_model.set_vocab.remote(vocab)
        self.labels = np.array(vocab)
        self.display_labels = np.array([vocab_map.get(x,x) for x in vocab])
        self.valid_labels = np.array([
            x not in self.IGNORE_VOCAB or x in vocab_map for x in vocab
        ])

    def predict2(self, im):
        im_id = ray.put(im)
        detic_result = self.detic_model.predict.remote(im_id)
        #egohos_result = self.egohos_model.predict_box.remote(im_id)
        (
            (xyxy, class_ids, confs, box_confs),
            #,
            #hoi_xyxy
        ) = ray.get([detic_result])#, egohos_result])

        # combine (exact) duplicate bounding boxes
        xyxy_unique, ivs = Detic.group_proposals(xyxy)
        # convert image boundaries to [0, 1]
        xyxyn_unique = boxnorm(xyxy_unique, *im.shape[:2])

        #if hoi_box is not None:
        #    ious = box_iou(hoi_box, torch.as_tensor(xyxy_unique))[0].numpy()
        #else:
        ious = np.zeros(len(xyxy_unique))

        return xyxyn_unique, ivs, confs, class_ids, box_confs, ious

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

    def run_offline(self, recording_id, recipe_id):
        from ptgprocess.record import RawReader, RawWriter, JsonWriter
        self.change_recipe(recipe_id)

        out_sids = ['detic:image', 'detic:image:v2', 'detic:image:for3d', 'detic:hands']

        raw_dir = os.path.join(RECORDING_RAW_DIR, recording_id)
        post_dir = os.path.join(RECORDING_POST_DIR, recording_id)
        print("raw directory:", raw_dir)
        with RawReader(os.path.join(raw_dir, 'main')) as reader, \
             RawWriter(out_sids[0], raw_dir) as writer0, \
             RawWriter(out_sids[1], raw_dir) as writer1, \
             RawWriter(out_sids[2], raw_dir) as writer2, \
             RawWriter(out_sids[3], raw_dir) as writer3, \
             JsonWriter(out_sids[0], post_dir) as json0, \
             JsonWriter(out_sids[1], post_dir) as json1, \
             JsonWriter(out_sids[2], post_dir) as json2, \
             JsonWriter(out_sids[3], post_dir) as json3:

            for ts, d in reader:
                # compute box
                im, im_params = self.load_image(d)
                (xyxy, ivs, confs, class_ids, box_confs, ious) = self.predict(im)

                objsv1 = self.as_v1_objs(xyxy, ivs, confs, class_ids)
                objsv2 = self.as_v2_objs(xyxy, ivs, confs, class_ids, box_confs, ious)
                hand = self.get_hand_presence(self.labels[class_ids], confs)
                obj_params = { 'objects': objsv1, 'image_params': im_params }

                writer0.write(jsondump(objsv1), ts)
                writer1.write(jsondump(objsv2), ts)
                writer2.write(jsondump(obj_params), ts)
                writer3.write(jsondump(hand), ts)
                json0.write(jsondump({'data': objsv1, 'timestamp': ts}), ts)
                json1.write(jsondump({'data': objsv2, 'timestamp': ts}), ts)
                json2.write(jsondump({'data': obj_params, 'timestamp': ts}), ts)
                json3.write(jsondump({'data': hand, 'timestamp': ts}), ts)




def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)



def boxnorm(xyxy, h, w):
    xyxy[:, 0] = (xyxy[:, 0]) / w
    xyxy[:, 1] = (xyxy[:, 1]) / h
    xyxy[:, 2] = (xyxy[:, 2]) / w
    xyxy[:, 3] = (xyxy[:, 3]) / h
    return xyxy


if __name__ == '__main__':
    import fire
    fire.Fire(DeticApp)
