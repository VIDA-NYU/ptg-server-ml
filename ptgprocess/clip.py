from __future__ import annotations
import os
import asyncio
import orjson
import tqdm

import numpy as np
from PIL import Image
import cv2

import ptgctl
from ptgctl import holoframe
from ptgctl.util import parse_epoch_time
from .core import Processor
from .util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_text_list



class ZeroClip(Processor):
    output_prefix = 'zero_clip'
    prompts = {
        'tools': 'a photo of a {}',
        'ingredients': 'a photo of a {}',
        'instructions': '{}',
    }
    STORE_DIR = 'post'

    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        import torch, clip
        self.device = device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.tokenize = clip.tokenize

    async def call_async(self, recipe_id, *a, **kw):
        return await asyncio.gather(
            self._call_async(recipe_id, *a, **kw),
            self._watch_server_state(recipe_id)
        )

    async def _watch_recipe_id(self, recipe_id):
        loop = asyncio.get_event_loop()
        while True:
            if recipe_id and self.current_id != recipe_id:
                break
            new_recipe_id, _ = await asyncio.gather(
                loop.run_in_executor(None, self.api.recipes.current),
                asyncio.sleep(3)
            )

    async def call_async(self, recipe_id, replay=None, fullspeed=None, out_file=None, fps=10, show=True, store_dir=None):
        self.current_id = recipe_id
        import torch
        assert recipe_id, "You must provide a recipe ID, otherwise we have nothing to compare"
        # load the recipe from the api
        recipe = self.api.recipes.get(recipe_id)
        texts = {k: recipe[k] for k, _ in self.prompts.items()}
        z_texts = {k: self.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

        if out_file is True:
            out_file = os.path.join(store_dir or self.STORE_DIR, replay or nowstring(), f'{self.output_prefix}.mp4')

        out_keys = set(texts)
        out_sids = [f'{replay or ""}{self.output_prefix}:{k}' for k in out_keys]
        async with StreamReader(self.api, ['main'], recording_id=replay, fullspeed=fullspeed) as reader, \
                   StreamWriter(self.api, out_sids, test=True) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async def _stream():
                async for _, t, d in reader:
                    if recipe_id and self.current_id != recipe_id:
                        print("recipe changed", recipe_id, '->', self.current_id)
                        break
                    # encode the image and compare to text queries
                    im = d['image']
                    z_image = self.encode_image(im)
                    sims = {k: self.compare_image_text(z_image, z_texts[k])[0] for k in out_keys}
                    # await writer.write([self._bundle(texts[k], sims[k]) for k in out_keys])
                    imout.output(draw_text_list(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [
                        f'{texts[k][i]} ({sims[k][i]:.0%})' 
                            for k in out_keys for i in torch.topk(sims[k], 3, dim=-1)[1].tolist()
                    ])[0], t)

            await asyncio.gather(_stream(), reader.watch_replay())

    def encode_text(self, texts, prompt_format=None, return_texts=False):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        toks = self.tokenize([prompt_format.format(x) for x in texts] if prompt_format else texts).to(self.device)
        z = self.model.encode_text(toks)
        z /= z.norm(dim=-1, keepdim=True)
        return (z, texts) if return_texts else z

    def encode_image(self, image):
        '''Encode image to CLIP embedding.'''
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        z_image = self.model.encode_image(image)
        z_image /= z_image.norm(dim=-1, keepdim=True)
        return z_image

    def compare_image_text(self, z_image, z_text):
        '''Compare image and text similarity (not sure why the 100, it's from the CLIP repo).'''
        return (100.0 * (z_image @ z_text.T)).softmax(dim=-1)

    def _bundle(self, text, similarity):
        '''Prepare text and similarity to be uploaded.'''
        return dict(zip(text, similarity.tolist()))


if __name__ == '__main__':
    import fire
    fire.Fire({
        'zero': ZeroClip,
    })