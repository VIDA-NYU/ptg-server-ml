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


holoframe_load = functools.lru_cache(maxsize=32)(holoframe.load)

class RecipeExit(Exception):
    pass

class Perception:
    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'perception',
                              password=os.getenv('API_PASS') or 'perception')

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
                    print(sid, ts, data, recipe_id)
                    if data != recipe_id:
                        return data

    async def start_session(self, recipe_id, prefix=None):
        '''Initialize the action session for a recipe.'''
        if isinstance(recipe_id, bytes):
            recipe_id = recipe_id.decode()
        if not recipe_id:
            raise RecipeExit("no recipe set.")
        self.session = StepsSession(prefix=prefix)
        await self.session.load_skill(recipe_id)

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        await self.start_session(recipe_id, prefix=prefix)

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

        t0 = time.time()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull:
            pbar = tqdm.tqdm()
            while True:
                pbar.set_description('read: waiting for data...')
                for sid, t, d in await ws_pull.recv_data():
                    pbar.set_description(f'read: {sid} {t}')
                    pbar.update()
                    try:
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            if time.time() - t0 < 1 and recipe_id == d.decode(): # HOTFIX: why does this happen?
                                continue
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            return 
                        queue.push([sid, t, holoframe_load(d)['image']])
                        await asyncio.sleep(1e-2)
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
                pbar.set_description(f'processor got {sid} {t}')
                pbar.update()
                xs = queue.read_buffer()
                imgs = [x for _, _, x in xs]

                # predict actions
                #for img in imgs:
                preds = await self.session.on_image(imgs)
                if preds is not None:
                        out_queue.push([preds, t])
            except Exception as e:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(1e-1)
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
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1e-1)
                finally:
                    queue.task_done()
    
async def ws_write_data_dict(ws_push, data, timestamp):
    data = {k: v for k, v in data.items() if v is not None}
    await ws_push.send_data(
        [jsondump(x) for x in data.values()], 
        list(data.keys()), 
        [noconflict_ts(timestamp)]*len(data))

def noconflict_ts(ts):
    return ts.split('-')[0] + '-*'

def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)



import ray
# import models
import cv2
import yaml
import functools
from step_recog.models import StepNet
from step_recog.statemachine import ProcedureStateMachine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@ray.remote(name='omnimix', num_gpus=1)
class AllInOneModel:
    def __init__(self, skill=None):
        cfg = yaml.load(open("procedural_step_recog/config/without_state_head.yaml"), Loader=yaml.CLoader)
        self.model = StepNet(cfg).eval().to(device)
        #path = '/src/app/models/best_model_new_cont.pth'
        path = '/src/app/models/best_model_focal_loss.pth'
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint)
        
        self.device = device #torch.cuda.current_device()
        skill and self.load_skill(skill)

    def load_skill(self, skill):
        self.skill = skill.upper()
        n = len(self.model.SKILL_STEPS[self.skill]) - 1
        self.sm = ProcedureStateMachine(n)
        self.model.STEPS = [f'{i}' for i in range(n)]
        self.buffer = None
        self.h = None

    def get_steps(self):
        return self.model.STEPS

    @torch.no_grad()
    def queue_frames(self, ims_bgr):
        for im in ims_bgr:
            self.model.queue_frame(im)

    def prepare_image(self, im, size=224, bgr2rgb = True):
        '''[H, W, 3] => [3, 224, 224]'''
        im = cv2.resize(im, (size, size))
        #if bgr2rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im.astype(float) / 255
        im = im.transpose(2, 0, 1)    
        return torch.Tensor(im)

    @torch.no_grad()
    def forward(self, ims_bgr):
        #print(ims_bgr.shape)
        ims_bgr = torch.stack([self.prepare_image(im) for im in ims_bgr])[None]
        self.buffer = self.model.insert_image_buffer(ims_bgr, self.buffer)[:, -32:]
        #print(self.buffer.shape)
        y_hat_steps, y_hat_state_machine, omni_outs, self.h = self.model(self.buffer.clone(), self.h)
        prob_step, prob_state = self.model.skill_step_proba(y_hat_steps, y_hat_state_machine, skill=self.skill)
        prob_step = prob_step[0, -1].cpu().numpy()
        try:
            self.sm.process_timestep(prob_step)
        except:
            print("\nerror state update", prob_step, self.sm.current_state)
        state = self.sm.current_state
        return prob_step, None, state

    @torch.no_grad()
    def forward_boxes(self, im_bgr):
        #im_rgb = im_bgr[:,:,::-1]  # from src: im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        #print(im_bgr.shape)
        box_results = self.model.yolo(im_bgr, verbose=False)
        objects = cvt_objects(box_results, self.model.OBJECT_LABELS)
        return objects

model = AllInOneModel.remote()




class StepsSession:
    MAX_RATE_SECS = 0.3
    def __init__(self, prefix=None):
        # models
        self.model = model

        # output names
        prefix = prefix or ""
        self.step_sid = f'{prefix}omnimix:step'
        self.step_id_sid = f'{prefix}omnimix:step:id'
        self.box_sid = f'{prefix}detic:image'
        self.steps_sid = f'{prefix}omnimix:steps:sm'
        self.steps_raw_sid = f'{prefix}omnimix:steps:sm:raw'
        self.out_sids = [
            self.step_sid,
            self.step_id_sid,
            self.box_sid,
            self.steps_sid,
            self.steps_raw_sid,
        ]

    async def load_skill(self, recipe_id):
        await self.model.load_skill.remote(recipe_id)
        vocab = await self.model.get_steps.remote()

        self.vocab = np.concatenate([np.asarray([f'{i+1}|{v}' for i, v in enumerate(vocab)]), np.array(['|OTHER'])])
        self.vocab_list = self.vocab.tolist()

        id_vocab = np.concatenate([np.arange(len(vocab)), np.array([-1])])
        self.id_vocab_list = id_vocab.astype(str).tolist()

        self.t0 = None

    async def on_image(self, imgs: List[np.ndarray]):
        if self.t0 is None:
            self.t0 = time.time()
        # if self.MAX_RATE_SECS > (time.time() - self.t0):
        #     self.model.queue_frames.remote(imgs)
        #     # objects = await self.model.forward_boxes.remote(imgs[-1])
        #     if not objects:
        #         return
        #     return { self.box_sid: objects }

        self.t0 = time.time()

        steps, objects, state = await self.model.forward.remote(imgs)
        if steps is None:
            return {}
        # convert everything to json-friendly
        step_list = steps.tolist()
        return {
            self.step_sid: dict(zip(self.vocab_list, step_list)),
            self.step_id_sid: dict(zip(self.id_vocab_list, step_list)),
            # self.box_sid: objects,#as_v1_objs(xyxyn, confs, class_ids, labels),
            self.steps_sid: {
                "all_steps": self.get_steps(state, step_list),
                "error_status": False,
                "error_description": "",
            },
            self.steps_raw_sid: state.tolist(),

        }

    def get_steps(self, state, confidence):
        return [
            {
                'number': i+1, 
                'name': f'{i+1}', # self.vocab[i], 
                'state': STATES[state[i]], 
                'confidence': confidence[i] if state[i]==1 else 1
            }
            for i in range(len(state))
        ]

STATES = ['unobserved', 'current', 'done']



if __name__ == '__main__':
    import fire
    fire.Fire(Perception)
