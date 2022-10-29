import os
import orjson
import asyncio
from collections import OrderedDict, defaultdict

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import numpy as np

import ptgctl
from ptgctl import holoframe
import ptgctl.util
from ptgprocess.egovlp import EgoVLP

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')



class EgoVLPApp:
    hands_threshold = 0.01
    action_sets = {
        'steps': {
            'prompt': '{}',
            'keys': ['steps_simple']
        }
    }

    def __init__(self, **kw):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'egovlp',
                              password=os.getenv('API_PASS') or 'egovlp')
        self.model = EgoVLP(**kw)

    @ptgctl.util.async2sync
    async def run(self, *a, **kw):
        while True:
            try:
                recipe_id = self.api.session.current_recipe()
                while not recipe_id:
                    print("waiting for recipe to be activated")
                    recipe_id = await self._watch_recipe_id(recipe_id)
                
                print("Starting recipe:", recipe_id)
                await self._run(recipe_id, *a, **kw)
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect(self.RECIPE_SID) as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    async def _run(self, recipe_id):
        # call the api and get the recipe vocab text
        self.set_recipe(recipe_id)

        # stream ids
        in_sids = ['main']
        recipe_sid = 'event:recipe:id'
        vocab_sid = 'event:vocab:update'
        hands_sid = 'detic:hands'

        out_keys = set(self.action_sets)
        out_sids = [f'egovlp:action:{k}' for k in out_keys]

        self.decay = 0.5
        self.sim_decay = {k: 0 for k in out_keys}
        self.hands = {}

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect(in_sids + [hands_sid, recipe_sid, vocab_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
                while True:
                    pbar.set_description('waiting for data...')
                    for sid, t, d in await ws_pull.recv_data():
                        pbar.set_description(f'{sid} {t}')
                        pbar.update()
                        tms = int(t.split('-')[0])

                        # watch recipe changes
                        if sid == recipe_sid:
                            print("recipe changed", recipe_id, '->', d, flush=True)
                            if not d: break
                            self.set_recipe(d.decode('utf-8'))
                            self.sim_decay = {k: 0 for k in out_keys}
                            continue
                        if not self.texts:
                            print("no text to compare to", flush=True)
                            return
                        # record changes in hand presence
                        if sid == hands_sid:
                            self.hands = orjson.loads(d)
                            self.hands['_timems'] = tms
                            continue
                        
                        # encode the image
                        d = holoframe.load(d)
                        im = d['image']
                        self.model.add_image(im)
                        z_image = self.model.predict_recent()
                    
                        # compare text and image
                        sims = {}
                        for k in out_keys:
                            sims[k] = self.compare(z_image, k)

                        # use hand presence as a gate
                        if self.hands.get('mean_conf_smooth', 1) < self.hands_threshold:
                            if tms - self.hands['_timems'] < 5*1000:
                                for v in sims.values():
                                    v[:] = 0
                            else:
                                self.hands = {}

                        # send data
                        await ws_push.send_data([
                            self.dump(self.texts[k], sims[k]) 
                            for k in out_keys
                        ], out_sids, [t for k in out_keys])

    def set_recipe(self, recipe_id):
        if not recipe_id:
            print('no recipe. using default vocab:', None)
            self.texts = self.z_texts = {}
            return
        print('using recipe', recipe_id)
        recipe = self.api.recipes.get(recipe_id)
        self.texts = {k: [t for k in d['keys'] for t in recipe.get(k) or []] for k, d in self.action_sets.items()}
        self.z_texts = {k: self.model.encode_text(self.texts[k], d['prompt']) for k, d in self.action_sets.items()}

    def compare(self, z_image, key):
        sim = self.model.similarity(self.z_texts[key], z_image)[0].detach()
        self.sim_decay[key] = sim = (1 - self.decay) * sim + self.decay * self.sim_decay[key]
        return sim

    def dump(self, text, similarity):
        return jsondump(dict(zip(text, similarity.tolist())))


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(EgoVLPApp)
