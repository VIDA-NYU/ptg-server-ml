from __future__ import annotations
import os
import asyncio
import orjson

import cv2
import numpy as np
import torch

from .core import Processor
from ..util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_text_list
from ..clip import ZeroClip, ActionClip1, ActionClip2

class AsyncExit(Exception): pass

class ZeroClipProcessor(Processor):
    output_prefix = 'clip:zero'
    prompts = {
        # 'tools': 'a photo of a {}',
        # 'ingredients': 'a photo of a {}',
        'steps': '{}',
    }
    STORE_DIR = 'post'

    Model = ZeroClip

    async def call_async(self, recipe_id=None, *a, **kw):
        recipe_id = self.api.sessions.current_recipe()
        while not recipe_id:
            print("waiting for recipe to be activated")
            recipe_id = await self._watch_recipe_id(recipe_id)
        await self._call_async(recipe_id, *a, **kw)
        return
        
        if recipe_id:
            print("explicitly set recipe ID", recipe_id)
            return await self._call_async(recipe_id, *a, **kw)







        recipe_id = self.api.sessions.current_recipe()
        while not recipe_id:
            print("waiting for recipe to be activated")
            recipe_id = await self._watch_recipe_id(recipe_id)
        print("Starting recipe:", recipe_id)
        
        t = asyncio.create_task(self._call_async(recipe_id, *a, **kw))
        try:
            await self._watch_recipe_id(recipe_id)
        finally:
            if not t.done():
                t.cancel()
            else:
                t.result()

    async def _watch_recipe_id(self, recipe_id):
        loop = asyncio.get_event_loop()
        while True:
            new_recipe_id, _ = await asyncio.gather(
                loop.run_in_executor(None, self.api.sessions.current_recipe),
                asyncio.sleep(3)
            )
            if new_recipe_id != recipe_id:
                self.current_id = new_recipe_id
                return new_recipe_id

    async def _call_async(self, recipe_id, replay=None, fullspeed=None, out_file=None, fps=5, show=False, test=False, store_dir=None, **kw):
        if not hasattr(self, 'model'):
            self.model = self.Model(**kw)
        self.current_id = recipe_id
        assert recipe_id, "You must provide a recipe ID, otherwise we have nothing to compare"
        # load the recipe from the api
        recipe = self.api.recipes.get(recipe_id)
        texts = {k: recipe[k] for k, _ in self.prompts.items()}
        z_texts = {k: self.model.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

        if out_file is True:
            out_file = os.path.join(
                store_dir or self.STORE_DIR, replay or nowstring(), 
                f'{self.__class__.__name__}-{self.output_prefix}.mp4')

        out_keys = set(texts)
        out_sids = [f'{replay or ""}{self.output_prefix}:{k}' for k in out_keys]
        async with StreamReader(self.api, ['main'], recording_id=replay, fullspeed=fullspeed) as reader, \
                   StreamWriter(self.api, out_sids, test=test) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async for _, t, d in reader:
                if recipe_id and self.current_id != recipe_id:
                    print("recipe changed", recipe_id, '->', self.current_id)
                    break
                # encode the image and compare to text queries
                im = d['image']
                z_image = self.model.encode_image(im)
                #if z_image is None:
                #    continue
                sims = {k: self.model.compare_image_text(z_image, z_texts[k])[0] for k in out_keys}
                
                await writer.write([
                    self.dump(texts[k], sims[k]) 
                    for k in out_keys
                ])

                if imout.active:
                    imout.output(draw_text_list(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [
                        f'{texts[k][i]} ({sims[k][i]:.0%})' 
                        for k in out_keys 
                        for i in torch.topk(sims[k], 3, dim=-1)[1].tolist()
                    ])[0], t)


    def dump(self, text, similarity):
        return orjson.dumps(
            dict(zip(text, similarity.tolist())), 
            option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


class ActionClip1Processor(ZeroClipProcessor):
    output_prefix = 'clip:action-mean'
    Model = ActionClip1


class ActionClip2Processor(ZeroClipProcessor):
    output_prefix = 'clip:action'
    Model = ActionClip2


if __name__ == '__main__':
    import fire
    fire.Fire({
        'zero': ZeroClipProcessor,
        'action1': ActionClip1Processor,
        'action2': ActionClip2Processor,
    })
