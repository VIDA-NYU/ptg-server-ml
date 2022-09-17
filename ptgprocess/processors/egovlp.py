from __future__ import annotations
import os
import asyncio
import orjson

import cv2
import numpy as np
import torch

from .core import Processor
from ..util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_text_list
from ..egovlp import EgoVLP


class EgoVLPApp(Processor):
    HANDS_SID = 'detic:hands'
    RECIPE_SID = 'event:recipe:id'
    output_prefix = 'egovlp:action'
    prompts = {
        # 'tools': 'a photo of a {}',
        # 'ingredients': 'a photo of a {}',
        'steps_simple': '{}',
    }
    key_map = {'steps_simple': 'steps'}
    STORE_DIR = 'post'

    HANDS_THRESHOLD = 0.01

    async def call_async(self, recipe_id=None, *a, **kw):
        if recipe_id:
            print("explicitly set recipe ID", recipe_id)
            return await self._call_async(recipe_id, *a, **kw)

        recipe_id = self.api.sessions.current_recipe()
        while not recipe_id:
            print("waiting for recipe to be activated")
            recipe_id = await self._watch_recipe_id(recipe_id)
        print("Starting recipe:", recipe_id)
        await self._call_async(recipe_id, *a, **kw)
        
    async def _watch_recipe_id(self, recipe_id):
        async with self.api.data_pull_connect(self.RECIPE_SID) as ws:
            while True:
                for sid, ts, data in (await ws.recv_data()):
                    if data != recipe_id:
                        return data

    def set_vocab(self, recipe_id):
        if not recipe_id:
            self.texts = self.z_texts = {}
            return
        recipe = self.api.recipes.get(recipe_id)
        self.texts = {k: recipe[k] for k, _ in self.prompts.items()}
        self.z_texts = {k: self.model.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

    async def _call_async(self, recipe_id, replay=None, fullspeed=None, decay=0.6, out_file=None, fps=5, show=False, test=False, store_dir=None, **kw):
        from ptgctl import holoframe
        if not hasattr(self, 'model'):
            self.model = EgoVLP(**kw)
        
        # get the vocab
        assert recipe_id, "You must provide a recipe ID, otherwise we have nothing to compare"
        self.set_vocab(recipe_id)

        self.hands = {}

        # optionally output to a file
        if out_file is True:
            out_file = os.path.join(
                store_dir or self.STORE_DIR, replay or nowstring(), 
                f'{self.__class__.__name__}-{self.output_prefix}.mp4')

        out_keys = set(self.texts)

        sim_decay = {k: 0 for k in out_keys}

        out_sids = [f'{replay or ""}{self.output_prefix}:{self.key_map.get(k, k)}' for k in out_keys]
        async with StreamReader(self.api, ['main', self.HANDS_SID], [self.RECIPE_SID], recording_id=replay, fullspeed=fullspeed, ack=True, raw=True) as reader, \
                   StreamWriter(self.api, out_sids, test=test) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async for sid, t, d in reader:
                # monitor the recipe
                if sid == self.RECIPE_SID:
                    print("recipe changed", recipe_id, '->', d, flush=True)
                    if not d: break
                    self.set_vocab(d.decode('utf-8'))
                    sim_decay = {k: 0 for k in out_keys}
                    continue
                if sid == self.HANDS_SID:
                    self.hands = orjson.loads(d)
                    continue
                if not self.texts:
                    print("no text to compare to", flush=True)
                    break

                # encode the image and compare to text queries
                d = holoframe.load(d)
                im = d['image']
                self.model.add_image(im)
                z_image = self.model.predict_recent()
                sims = {
                    k: (1 - decay) * self.model.similarity(self.z_texts[k], z_image)[0].detach() 
                       + decay * sim_decay[k]
                    for k in out_keys}

                if self.hands.get('mean_conf_smooth', 1) < self.HANDS_THRESHOLD:
                    for v in sims.values():
                        v[:] = 0
                
                # output
                await writer.write([self.dump(self.texts[k], sims[k]) for k in out_keys])
                if imout.active:
                    imout.output(draw_text_list(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [
                        f'{self.texts[k][i]} ({sims[k][i]:.0%})' for k in out_keys 
                        for i in torch.topk(sims[k], 3, dim=-1)[1].tolist()
                        if sims[k][i] > 0
                    ])[0], t)


    def dump(self, text, similarity):
        return orjson.dumps(
            dict(zip(text, similarity.tolist())), 
            option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


if __name__ == '__main__':
    import fire
    fire.Fire(EgoVLPApp)
