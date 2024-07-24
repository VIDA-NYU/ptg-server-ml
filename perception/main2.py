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

ray.init(num_gpus=2)


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
        self.session = MultiStepsSession()
        while True:
            try:
                await self.run_loop(*a, **kw)
            except RecipeExit as e:
                print(e)
                await asyncio.sleep(5)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def run_loop(self):
        # stream ids
        with logging_redirect_tqdm():
            async with ag.Graph() as g:
                q_rgb, = g.add_producer(self.reader, g.add_queue(ag.SlidingQueue))
                q_proc = g.add_consumer(self.processor, q_rgb, g.add_queue(ag.SlidingQueue))
                g.add_consumer(self.writer, q_proc)
            print("finished")

    async def reader(self, queue, prefix=None):
        # stream ids
        in_sid = f'{prefix or ""}main'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        t0 = time.time()
        async with self.api.data_pull_connect([in_sid, recipe_sid, vocab_sid], ack=True) as ws_pull:
            recipe_id = self.api.session.current_recipe()
            if recipe_id is not None:
                print('recipe id', recipe_id)
                await self.session.load_skill(recipe_id)
            pbar = tqdm.tqdm()
            while True:
                pbar.set_description('read: waiting for data...')
                for sid, t, d in await ws_pull.recv_data():
                    pbar.set_description(f'read: {sid} {t}')
                    pbar.update()
                    try:
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            #if time.time() - t0 < 1 and recipe_id == d.decode(): # HOTFIX: why does this happen?
                            #    continue
                            d = d.decode() if d else None
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            recipe_id = d if d else None
                            if recipe_id:
                                print('recipe id', recipe_id)
                                await self.session.load_skill(recipe_id)
                            continue
                        if recipe_id is None:
                            continue
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
        print("processor done")

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
        print("writer done")
    
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
import models
import functools
from step_recog.full.model import StepPredictor
from step_recog.full.statemachine import ProcedureStateMachine

@functools.lru_cache(3)
def get_model(skill, device):
    from step_recog.full.model import StepPredictor
    return StepPredictor(skill).to(device)



def cvt_objects(outputs, labels):
    boxes = outputs[0].boxes.cpu()
    objects = as_v1_objs(
           boxes.xyxyn.numpy(),
           boxes.conf.numpy(),
           boxes.cls.numpy(),
           labels[boxes.cls.int().numpy()],
           conf_threshold=0.5)

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


@ray.remote(num_gpus=1)
class AllInOneModel:
    def __init__(self, skill=None, model_type=None):
        self.model_type = model_type
        self.MODEL_CACHE = {}
        #self.device = 'cuda'#torch.cuda.current_device()
        self.device = torch.cuda.current_device()
        print(self.device)
        try:
            print("Preloading cache")
            for s in getattr(StepPredictor, 'PRELOAD_SKILLS', ['M2','M5','R18']):
                print("Preloading", s)
                self.load_skill(s)
            #print("Preloaded.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('Failed to preload models:', type(e), e)
        skill and self.load_skill(skill)

    def get_model(self, skill):
        skill = skill.upper()
        if skill not in self.MODEL_CACHE:
            self.MODEL_CACHE[skill] = StepPredictor(skill.upper()).to(self.device)
        model = self.MODEL_CACHE[skill]
        model.reset()
        model.eval()
        
        # y=model.yolo
        # model.yolo = None
        # model.eval()
        # model.head.eval()
        # model.omnivore.eval()
        # model.yolo=y
        return model

    def load_skill(self, skill):
        print("loading skill", skill)
        skill = skill.decode() if isinstance(skill, bytes) else skill or None
        self.model = self.get_model(skill)
        self.sm = ProcedureStateMachine(len(self.model.STEPS))
        print(555, skill, self.model.cfg.MODEL.OUTPUT_DIM, len(self.model.STEPS), self.model.STEPS)
        self.model.h = None
        print(666, skill, self.model.h is None, self.sm.current_state)

    def get_steps(self):
        return self.model.STEPS

    @torch.no_grad()
    def queue_frames(self, ims_bgr):
        for im in ims_bgr:
            self.model.queue_frame(im)

    @torch.no_grad()
    def forward(self, ims_bgr):
        #print(ims_bgr.shape)
        ims_bgr = [im[:,:,::-1] for im in ims_bgr]
        for im in ims_bgr[:-1]:
            self.model.queue_frame(im)
        if not self.model.has_omni_maxlen():
            return None, None, None
        preds, box_results = self.model(ims_bgr[-1], return_objects=True)
        objects = cvt_objects(box_results, self.model.OBJECT_LABELS)
        try:
            self.sm.process_timestep(preds.cpu().squeeze().numpy())
        except Exception as e:
            print("\nerror state update", preds, self.sm.current_state, type(e).__name__, e)
        state = self.sm.current_state
        return preds, objects, state

    @torch.no_grad()
    def forward_boxes(self, im_bgr):
        #im_rgb = im_bgr[:,:,::-1]  # from src: im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        #print(im_bgr.shape)
        box_results = self.model.yolo(im_bgr, verbose=False)
        objects = cvt_objects(box_results, self.model.OBJECT_LABELS)
        return objects

model = AllInOneModel.remote()
model2 = AllInOneModel.remote()

MODELS = {
    None: model,
    'model2': model2,
}


class StepsSession:
    MAX_RATE_SECS = 0.3
    def __init__(self, prefix=None, model=model):
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
        print(prefix, self.out_sids)

    async def load_skill(self, recipe_id):
        print(recipe_id)
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
        if self.MAX_RATE_SECS > (time.time() - self.t0):
            self.model.queue_frames.remote(imgs)
            objects = await self.model.forward_boxes.remote(imgs[-1])
            if not objects:
                return
            return { self.box_sid: objects }

        self.t0 = time.time()

        steps, objects, state = await self.model.forward.remote(imgs)
        if steps is None:
            return {}
        # convert everything to json-friendly
        step_list = steps[0].detach().cpu().tolist()
        return {
            self.step_sid: dict(zip(self.vocab_list, step_list)),
            self.step_id_sid: dict(zip(self.id_vocab_list, step_list)),
            self.box_sid: objects,#as_v1_objs(xyxyn, confs, class_ids, labels),
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



class MultiStepsSession:
    def __init__(self, models=MODELS):
        self.sessions = []
        for prefix, model in models.items():
            self.sessions.append(StepsSession(prefix, model))
        s = self.sessions[0]
        self.out_sids = s.out_sids

    async def load_skill(self, recipe_id):
        await asyncio.gather(*[s.load_skill(recipe_id) for s in self.sessions])

    async def on_image(self, imgs):
        data = {}
        ds = await asyncio.gather(*[s.on_image(imgs) for s in self.sessions])
        for d in ds:
            data.update(d or {})
        return data

    #def get_steps(self, state, confidence, i=0):
    #    return self.sessions[i].get_steps(state, confidence)



if __name__ == '__main__':
    import fire
    fire.Fire(Perception)
