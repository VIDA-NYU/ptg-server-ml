from __future__ import annotations
import os
import asyncio
import orjson

import cv2
import numpy as np
import torch

from .core import Processor
from ..util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_text_list
from ..omnivore import Omnivore


class OmnivoreProcessor(Processor):
    # def set_vocab(self, recipe_id):
    #     if not recipe_id:
    #         self.texts = self.z_texts = {}
    #         return
    #     recipe = self.api.recipes.get(recipe_id)
    #     self.texts = {k: recipe[k] for k, _ in self.prompts.items()}
    #     self.z_texts = {k: self.model.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

    async def call_async(self, replay=None, fullspeed=None, out_file=None, fps=5, show=False, test=False, store_dir=None, **kw):
        from ptgctl import holoframe

        if not hasattr(self, 'model'):
            self.model = Omnivore(**kw)

        # optionally output to a file
        if out_file is True:
            out_file = os.path.join(
                store_dir or self.STORE_DIR, replay or nowstring(), 
                f'{self.__class__.__name__}-{self.output_prefix}.mp4')

        async with StreamReader(self.api, ['main'], recording_id=replay, fullspeed=fullspeed, ack=True, raw=True) as reader, \
                   StreamWriter(self.api, ['omnivore:actions'], test=test) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async for sid, t, d in reader:
                # monitor the recipe
                if sid == self.RECIPE_SID:
                    print("recipe changed", recipe_id, '->', d, flush=True)
                    if not d: break
                    self.set_vocab(d.decode('utf-8'))
                    continue
                if not self.texts:
                    print("no text to compare to", flush=True)
                    break

                # encode the image and compare to text queries
                d = holoframe.load(d)
                im = d['image']
                y_pred = self.model(im)
                outputs, _ = self.model.topk(y_pred)

                # output
                await writer.write([self.dump(outputs)])
                if imout.active:
                    imout.output(draw_text_list(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), outputs, t))


    def dump(self, text):
        return orjson.dumps(text, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


if __name__ == '__main__':
    import fire
    fire.Fire(OmnivoreProcessor)
