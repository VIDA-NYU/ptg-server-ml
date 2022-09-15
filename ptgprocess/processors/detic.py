from __future__ import annotations
import os
import asyncio
import orjson
import tqdm

import cv2
import numpy as np

from ptgctl import holoframe
from ptgctl.util import parse_epoch_time
from ptgctl.pt3d import Points3D
from .core import Processor
from ..util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_boxes, call_multiple_async
from ..detic import Detic


DEFAULT_VOCAB = 'lvis'


class DeticApp(Processor):
    image_box_keys = ['xyxyn', 'confidence', 'class_id', 'label']
    STORE_DIR = 'post'
    vocab_keys = ['tools_simple', 'ingredients_simple']

    def __init__(self):
        super().__init__()
        self.model = Detic()

    async def call_async(self, replay=None, prefix=None, fullspeed=None, out_file=None, fps=10, show=False, store_dir=None, test=False, **kw):
        # stream ids
        in_sids = ['main']
        out_sids = ['detic:image']
        recipe_sid = 'event:recipe:id'

        # optional output video
        if out_file is True:
            out_file = os.path.join(store_dir or self.STORE_DIR, replay or nowstring(), f'{self.output_prefix}.mp4')

        async with StreamReader(self.api, in_sids, [recipe_sid], recording_id=replay, fullspeed=fullspeed, raw_ts=True, ack=True, **kw) as reader, \
                   StreamWriter(self.api, out_sids, test=test) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            
            # set initial recipe
            self.change_recipe(self.api.sessions.current_recipe())

            async for sid, t, x in reader:
                # watch for recipe changes
                if sid == recipe_sid:
                    await writer.write(b'[]', b'[]')
                    self.change_recipe(x)
                    continue
                
                im = x['image']
                xyxy, confs, class_ids, labels = self.predict(im)
                xywhn = xyxy.copy()
                h, w = im.shape[:2]
                xywhn[:, 0] = (xywhn[:, 0]) / w
                xywhn[:, 1] = (xywhn[:, 1]) / h
                xywhn[:, 2] = (xywhn[:, 2]) / w
                xywhn[:, 3] = (xywhn[:, 3]) / h
                
                # write back to stream
                await writer.write([
                    self.dump(
                        self.image_box_keys, 
                        [xywhn, confs, class_ids, labels]),
                ], out_sids, [t])

                # optional video display
                if imout.active:
                    imout.output(draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), xyxy, [
                        f'{l} {c:.0%} [{x:.0f},{y:.0f},{z:.0f}]' 
                        for l, c, (x,y,z) in zip(labels[valid], confs[valid], xyz_center[valid])
                    ]), parse_epoch_time(mts))

    def change_recipe(self, recipe_id):
        if not recipe_id:
            print('no recipe. using default vocab:', DEFAULT_VOCAB)
            self.model.set_vocab(DEFAULT_VOCAB)
            return
        print('using recipe', recipe_id)
        recipe = self.api.recipes.get(recipe_id)
        self.model.set_vocab([w for k in self.vocab_keys for w in recipe[k]])

    def predict(self, im):
        outputs = self.model(im)
        insts = outputs["instances"].to("cpu")
        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        labels = self.model.labels[class_ids]
        return xyxy, confs, class_ids, labels

    def dump(self, keys, xs):
        return jsondump([dict(zip(keys, xs)) for xs in zip(*xs)])


def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)


if __name__ == '__main__':
    import fire
    fire.Fire(DeticApp)
