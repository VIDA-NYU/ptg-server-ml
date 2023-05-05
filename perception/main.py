'''This is a test of perception utilizing the model server

'''
import os
import time
from typing import List
import asyncio
import orjson
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
import torch

import ray
import asyncio_graph as ag

import functools
import ptgctl
from ptgctl import holoframe

ray.init(num_gpus=1)

from steps_list import RECIPE_STEP_LABELS, RECIPE_STEP_IDS

holoframe_load = functools.lru_cache(maxsize=32)(holoframe.load)

class RecipeExit(Exception):
    pass

class Perception:
    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'perception',
                              password=os.getenv('API_PASS') or 'perception')
        # self.model = Omnimix().cuda()
        # self.model.eval()  # LOL fuck me. forgetting this is a bad time

    @torch.no_grad()
    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling and recipe status watching.'''
        while True:
            try:
                recipe_id = self.api.session.current_recipe()
                while not recipe_id:
                    print("waiting for recipe to be activated")
                    recipe_id = await self._watch_recipe_id(recipe_id)
                
                print("Starting recipe:", recipe_id)
                await self.run_recipe(recipe_id, *a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect('event:recipe:id') as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    def start_session(self, recipe_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if not recipe_id:
            raise RecipeExit("no recipe set.")
        # recipe = self.api.recipes.get(recipe_id) or {}
        # vocab = recipe.get(self.vocab_key)
        # step_map = recipe.get('step_map') or {}
        # if not vocab:
        #     raise Exception(f"\nRecipe {recipe_id}.{self.vocab_key} returned no steps in {set(recipe)}.")
        if recipe_id not in RECIPE_STEP_LABELS:
            raise RecipeExit(f"{recipe_id} not supported by this model.")
        vocab = RECIPE_STEP_LABELS[recipe_id]
        id_vocab = RECIPE_STEP_IDS[recipe_id]
        self.session = StepsSession(recipe_id, vocab, id_vocab, prefix=prefix)

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id, prefix=prefix)

        # stream ids
        with logging_redirect_tqdm():
            async with ag.Graph() as g:
                q_rgb, = g.add_producer(
                    self.reader, g.add_queue(ag.SlidingQueue), 
                    recipe_id=recipe_id, prefix=prefix)
                q_proc = g.add_consumer(self.processor, q_rgb, g.add_queue(ag.SlidingQueue))
                g.add_consumer(self.writer, q_proc)
            print("finished")


    async def reader(self, queue, recipe_id, prefix=None):
        # stream ids
        in_sid = f'{prefix or ""}main'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull:
            pbar = tqdm.tqdm()
            while True:
                pbar.set_description('read: waiting for data...')
                for sid, t, d in await ws_pull.recv_data():
                    pbar.set_description(f'read: {sid} {t}')
                    pbar.update()
                    # watch recipe changes
                    if sid == recipe_sid or sid == vocab_sid:
                        print("recipe changed", recipe_id, '->', d, flush=True)
                        return 
                    queue.push([sid, t, holoframe_load(d)['image']])
                    await asyncio.sleep(1e-2)

    async def processor(self, queue, out_queue):
        pbar = tqdm.tqdm()
        while True:
            pbar.set_description('processor waiting for data...')
            sid, t, d = await queue.get()
            try:
                pbar.set_description(f'processor got {sid} {t}')
                pbar.update()
                xs = queue.read_buffer()
                imgs = [x for _, _, x in xs]

                # predict actions
                preds = await self.session.on_image(imgs)
                if preds is not None:
                    out_queue.push([preds, t])
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
            finally:
                queue.task_done()


    async def writer(self, queue):
        '''Run the recipe.'''
        async with self.api.data_push_connect(self.session.out_sids, batch=True) as ws_push:
            # pbar = tqdm.tqdm()
            while True:
                # pbar.set_description('writer waiting for data...')
                preds, timestamp = await queue.get()
                try:
                    # pbar.set_description(f'writer got {set(preds)} {timestamp}')
                    # pbar.update()
                    await ws_write_data_dict(ws_push, preds, timestamp)
                finally:
                    queue.task_done()
    
async def ws_write_data_dict(ws_push, data, timestamp):
    data = {k: v for k, v in data.items() if v is not None}
    await ws_push.send_data(
        [jsondump(x) for x in data.values()], 
        list(data.keys()), 
        [noconflict_ts(timestamp)]*len(data))
        

import models

class FakeModel(torch.nn.Module):
    def forward(self, z_omni, z_clip, z_clip_patch, hidden):
        print(z_omni.shape, z_clip.shape, z_clip_patch.shape)
        x = z_clip * z_clip_patch.sum(0, keepdims=True)
        return x, hidden


# omnivore = models.OmnivoreModel.remote() #ray.get_actor("SERVE_REPLICA::omnivore")
# yolo = models.BBNYoloModel.remote() #ray.get_actor(f"bbn_yolo_{recipe_id}")
# clip_patches = models.ClipPatchModel.remote() #ray.get_actor("clip_patches")
# omnimix = models.OmnimixModel.remote()
model = models.AllInOneModel.remote()

class StepsSession:
    MAX_RATE_SECS = 0.5
    def __init__(self, recipe_id, vocab, id_vocab, prefix=None):
        self.vocab = np.asarray(vocab)
        self.vocab_list = self.vocab.tolist()
        self.id_vocab_list = [str(x) for x in np.asarray(id_vocab).tolist()]

        # models
        self.model = model
        model.load_skill.remote(recipe_id)

        # output names
        prefix = prefix or ""
        self.step_sid = f'{prefix}omnimix:step'
        self.step_id_sid = f'{prefix}omnimix:step:id'
        self.box_sid = f'{prefix}detic:image'
        self.out_sids = [
            self.step_sid,
            self.step_id_sid,
            self.box_sid,
        ]
        self.t0 = None
        self.hidden = None

    async def on_image(self, imgs: List[np.ndarray]):
        if self.t0 is None:
            self.t0 = time.time()
        if self.MAX_RATE_SECS > (time.time() - self.t0):
            objects = await self.model.forward_boxes.remote(imgs[-1])
            return { self.box_sid: objects }

        self.t0 = time.time()

        steps, objects, self.hidden = await self.model.forward.remote(imgs, self.hidden)
        # convert everything to json-friendly
        step_list = steps.detach().cpu().tolist()
        return {
            self.step_sid: dict(zip(self.vocab_list, step_list)),
            self.step_id_sid: dict(zip(self.id_vocab_list, step_list)),
            self.box_sid: objects,#as_v1_objs(xyxyn, confs, class_ids, labels),
        }

    # async def on_image(self, imgs: List[np.ndarray]):
    #     print(len(imgs), flush=True)
    #     # request input features
    #     future_omni = self.omnivore.encode_images.remote(imgs)
    #     # print(future_omni)
    #     future_boxes = self.yolo.forward.remote(imgs[-1])
    #     # print(future_boxes)
    #     future_patches = self.clip_patches.forward.remote(
    #         imgs[-1], get_boxes_from_dict.remote(future_boxes))
    #     # print(future_patches)
    #     # get result
    #     z_omni, z_clip, boxes = await asyncio.gather(
    #         future_omni, future_patches, future_boxes)
    #     print(len(z_omni), len(z_clip), len(boxes))

    #     # pull out box
    #     confs = np.array(boxes['confidence'])
    #     class_ids = np.array(boxes['class_ids'])
    #     labels = np.array(boxes['labels'])
    #     xywhn = np.array(boxes['xywhn'])
    #     xyxyn = xywhn.copy()
    #     xyxyn[:, 2:] -= xyxyn[:, :2]

    #     # join object features with box features
    #     z_clip_frame = np.concatenate([z_clip[:1], np.array([[0, 0, 1, 1, 1]])], axis=1)
    #     z_clip_patch = np.concatenate([z_clip[1:], xywhn, confs[:, None]], axis=1)

    #     # # compute step
    #     steps_output, self.hidden = self.omnimix.remote(
    #         [
    #             z_omni[None,None], 
    #             z_clip_frame[None,None], 
    #             z_clip_patch[None,None],
    #         ], 
    #         hidden=self.hidden)
    #     steps = steps_output[:,:,-2:]
    #     steps = z_clip_frame
        
    #     # convert everything to json-friendly
    #     step_list = steps.detach().cpu().tolist()
    #     return {
    #         self.step_sid: dict(zip(self.vocab_list, step_list)),
    #         self.step_id_sid: dict(zip(self.id_vocab_list, step_list)),
    #         self.box_sid: as_v1_objs(xyxyn, confs, class_ids, labels),
    #     }

@ray.remote
def get_boxes_from_dict(boxes):
    return boxes['xywhn']

def as_v1_objs(xyxy, confs, class_ids, labels, conf_threshold=0.1):
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


def noconflict_ts(ts):
    return ts.split('-')[0] + '-*'


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(Perception)
