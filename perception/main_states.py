'''This is a test of perception utilizing the model server

'''
import os
import time
from collections import defaultdict
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
from ptgctl.util import parse_epoch_time

from object_states.inference import Perception

ray.init(num_gpus=1)

from steps_list import RECIPE_STEP_LABELS, RECIPE_STEP_IDS

holoframe_load = functools.lru_cache(maxsize=8)(holoframe.load)

class RecipeExit(Exception):
    pass



class APIEvents:
    def __init__(self):
        self.callbacks = defaultdict(lambda: [])

    def add_callback(self, stream_id, func):
        self.callbacks[stream_id].append(func)

    async def __call__(self, stream_id, message, timestamp, **kw):
        all_messages = {}
        results = await asyncio.gather(*(f(message, timestamp, **kw) for f in self.callbacks[stream_id]))
        for r in results:
            all_messages.update(r or {})
        return all_messages



class APILoop:
    def __init__(self, api, input_sids, output_sids):
        self.api = api
        self.input_sids = input_sids
        self.output_sids = output_sids
        self.events = APIEvents()

    async def run(self):
        '''Run producer/consumer loop.'''
        async with ag.Graph() as g:
            q_msg, = g.add_producer(self.reader, g.add_queue(ag.SlidingQueue))
            q_proc = g.add_consumer(self.processor, q_msg, g.add_queue(ag.SlidingQueue))
            g.add_consumer(self.writer, q_proc)

    async def reader(self, queue):
        t0 = time.time()
        async with self.api.data_pull_connect(self.input_sids or '*', ack=True) as ws_pull:
            pbar = tqdm.tqdm()
            while True:
                pbar.set_description('read: waiting for data...')
                for sid, t, d in await ws_pull.recv_data():
                    pbar.set_description(f'read: {sid} {t}')
                    pbar.update()
                    try:
                        queue.push([sid, t, d])
                        await asyncio.sleep(0.05)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        await asyncio.sleep(1e-1)

    async def processor(self, queue, out_queue):
        pbar = tqdm.tqdm()
        while True:
            pbar.set_description('processor waiting for data...')
            sid, t, d = await queue.get()
            try:
                xs = queue.read_buffer()
                pbar.set_description(f'processor got {sid} {t} {len(xs)}')
                # predict actions
                for sid, t, d in xs:
                    pbar.update()
                    preds = await self.events(sid, d, t)
                    if preds:
                        out_queue.push([preds, '*'])
            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1e-1)
            finally:
                queue.task_done()

    async def writer(self, queue):
        async with self.api.data_push_connect(self.output_sids or '*', batch=True) as ws_push:
            while True:
                preds, timestamp = await queue.get()
                try:
                    await ws_write_data_dict(ws_push, preds, timestamp)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1e-1)
                finally:
                    queue.task_done()



VOCAB = {
    # 'base': 'lvis',
    'tracked': [
        # "tortilla",
        'tortilla pizza plain circular paper_plate quesadilla pancake: tortilla',
        # 'tortilla pizza plain circular paper_plate: tortilla',
        "mug coffee tea: mug",
        "bowl soup_bowl: bowl",
        "microwave_oven",
        "plate",

    ],
    'untracked': [
        "tortilla plastic_bag packet ice_pack circular: tortilla_package",
        'banana',
        "banana mushroom: banana_slice",
        'chopping_board clipboard place_mat tray: cutting_board',
        'knife',
        'jar bottle can: jar',
        'jar_lid bottle_cap: jar_lid',
        'toothpicks',
        # 'floss',
        'watch', 'glove', 'person',
    ],
    'equivalencies': {
        # equivalencies
        # 'can': 'bottle',
        'beer_can': 'bottle',
        'clipboard': 'chopping_board',
        'place_mat': 'chopping_board',
        'tray': 'chopping_board',
        
        # labels to ignore
        'table-tennis_table': 'IGNORE', 
        'table': 'IGNORE', 
        'dining_table': 'IGNORE', 
        'person': 'IGNORE',
        'watch': 'IGNORE',
        'glove': 'IGNORE',
        'magnet': 'IGNORE',
        'vent': 'IGNORE',
        'crumb': 'IGNORE',
        'nailfile': 'IGNORE',

        # not sure
        'handle': 'IGNORE',
    }
}

@ray.remote(num_gpus=1)
class PerceptionAgent:
    def __init__(self, vocab, state_db, detect_every):
        self.perception = Perception(
            vocabulary=vocab,
            state_db_fname=state_db,
            detect_every_n_seconds=detect_every,
            # detic_device='cuda:1',
            # egohos_device='cuda:0',
            # xmem_device='cuda:1',
            # clip_device='cuda:1',
        )

    def predict(self, frame, timestamp):
        t0 = time.time()
        track_detections, frame_detections, hoi_detections = self.perception.predict(frame, timestamp)
        return {
            "detic:image": self.perception.serialize_detections(track_detections, frame.shape),
        }

class PerceptionApp:
    def __init__(self, vocab=VOCAB, state_db='v0', detect_every=1, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'perception',
                              password=os.getenv('API_PASS') or 'perception')
        self.loop = APILoop(self.api, ['main'], ['detic:image'])
        self.agent = PerceptionAgent.remote(vocab, state_db, detect_every)

        self.loop.events.add_callback('main', self.on_image)
        self.loop.events.add_callback('task:control', self.on_control)

    async def on_image(self, message, timestamp):
        frame = holoframe_load(message)['image'][:,:,::-1]
        timestamp = parse_epoch_time(timestamp)
        return await self.agent.predict.remote(frame, timestamp)

    async def on_control(self, message, timestamp):
        # self.perception.detector.xmem.clear_memory()
        pass

    @torch.no_grad()
    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        '''Persistent running app, with error handling.'''
        while True:
            try:
                await self.loop.run(*a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)


async def ws_write_data_dict(ws_push, data, timestamp):
    data = {k: v for k, v in data.items() if v is not None}
    await ws_push.send_data(
        [jsondump(x) for x in data.values()], 
        list(data.keys()), 
        [noconflict_ts(timestamp)]*len(data))


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
    return '*' if ts == '*' else ts.split('-')[0] + '-*'


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(PerceptionApp)
