import os
import orjson
import asyncio
from collections import OrderedDict, defaultdict, deque

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np
from torch import nn
from torch.nn import functional as F

import ptgctl
from ptgctl import holoframe
import ptgctl.util

from ptgprocess.omnimix import Omnivore, AudioSlowFast, OmniGRU

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

import torch
# from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StepsSession:
    vocab_key = 'steps_simple'
    decay = 0.5
    def __init__(self, model, vocab, step_mask, prefix=None):
        self.model = model
        self.rgb_q = deque(maxlen=32)
        self.aud_q = deque(maxlen=16)
        self.rgb_emb_q = deque(maxlen=8)
        self.aud_emb_q = deque(maxlen=8)
        self.step_vocab = vocab
        self.step_mask = torch.Tensor(step_mask).int()
        self.sim_decay = 0
        prefix = f'{prefix or ""}omnimix'

        self.step_sid = f'{prefix}:step'
        self.rgb_verb_sid = f'{prefix}:verb:rgb'
        self.audio_verb_sid = f'{prefix}:verb:audio'
        self.output_sids = [
            self.step_sid,
            # f'{prefix}:action:rgb',
            self.rgb_verb_sid,
            # f'{prefix}:noun:rgb',
            self.audio_verb_sid,
            # f'{prefix}:noun:audio',
        ]
        self.empty_rgb_emb = torch.zeros(1, 1024).to(device)
        self.empty_aud_emb = torch.zeros(1, 2304).to(device)

    def format_message(self, steps):
        return {
            "populated": True,
            "casualty currently working on": {
                "casualty": 1,
                "confidence": 1.0,
            },
            "current skill": {
                "number": "R18",
                "confidence": 1.0,
            },
            "users current actions right now": {
                "steps": [
                    {
                        "name": name,
                        "state": "idk",
                        "confidence": conf,
                    }
                    for name, conf in steps.items()
                ],
            }
        }

    def on_image(self, image, **extra):
        # add the image to the frame queue
        self.rgb_q.append(self.model.rgb.prepare_image(image))
        verb, noun = self._predict_video()
        steps = self._predict_step()
        return {
            self.step_sid: dict(zip(self.step_vocab.tolist(), steps.tolist())) if steps is not None else None, 
            self.rgb_verb_sid: dict(zip(self.model.aud.vocab[0], verb[0].tolist())),
        }


class RecipeExit(Exception):
    pass



class MsgApp:
    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'bbn_msgs',
                              password=os.getenv('API_PASS') or 'bbn_msgs')

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
        self.session = MsgSession()

    async def run_recipe(self, recipe_id, prefix=None):
        '''Run the recipe.'''
        self.start_session(recipe_id, prefix=prefix)

        # stream ids
        # in_rgb_sid = 'main'
        # in_aud_sid = 'mic0'
        in_rgb_sid = f'{prefix or ""}main'
        in_aud_sid = f'{prefix or ""}mic0'
        recipe_sid = f'{prefix or ""}event:recipe:id'
        vocab_sid = f'{prefix or ""}event:recipes'

        print(self.session.output_sids)

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect([in_rgb_sid, in_aud_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(self.session.output_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        # watch recipe changes
                        if sid == recipe_sid or sid == vocab_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            self.start_session(recipe_id, prefix=prefix)
                            continue

                        # predict actions
                        preds = None
                        if sid == in_rgb_sid:
                            preds = self.session.on_image(**holoframe.load(d))
                        elif sid == in_aud_sid:
                            preds = self.session.on_audio(**holoframe.load(d))
                        if preds is not None:
                            preds = {k: v for k, v in preds.items() if v is not None}
                            await ws_push.send_data(
                                [jsondump({k: round(v, 2) for k, v in sorted(x.items(), key=lambda x: -x[1])}) for x in preds.values()], 
                                list(preds.keys()), 
                                [t]*len(preds))

    # def run_offline(self, recording_id, recipe_id, out_sid='egovlp:action:steps'):
    #     from ptgprocess.record import RawReader, RawWriter, JsonWriter
    #     self.start_session(recipe_id)

    #     raw_dir = os.path.join(RECORDING_RAW_DIR, recording_id)
    #     post_dir = os.path.join(RECORDING_POST_DIR, recording_id)
    #     print("raw directory:", raw_dir)
    #     #  RawWriter(out_sid, raw_dir) as writer, 
    #     with RawReader(os.path.join(raw_dir, 'main')) as reader, \
    #          JsonWriter(out_sid, post_dir) as json_writer:
 
    #         for ts, d in reader:
    #             sim_dict = self.session.on_image(**holoframe.load(d))
    #             writer.write(jsondump(sim_dict), ts)
    #             json_writer.write(jsondump({**sim_dict, 'timestamp': ts}), ts)




def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(StepsApp)
