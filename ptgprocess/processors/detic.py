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


DEFAULT_VOCAB = [
    'laptop', 'water bottle',
]


class Detic3D(Processor):
    output_prefix = 'detic'
    image_box_keys = ['xywhn', 'confidence', 'class_id', 'labels']
    world_box_keys = ['xyz_center', 'xyz_top', 'confidence', 'class_id', 'labels']
    min_dist_secs = 1
    max_depth_dist = 7
    STORE_DIR = 'post'

    def __init__(self):
        super().__init__()
        self.model = Detic()

    async def call_async(self, replay=None, fullspeed=None, prefix=None, out_file=None, fps=10, show=False, store_dir=None, test=False, **kw):
        # initial state management
        self.data = {}
        self.ts = {}
        self.model_results = None

        # stream ids
        prefix = prefix or ''
        out_prefix = f'{prefix}{self.output_prefix}'
        self.in_sids = in_sids = ['main', 'depthlt', 'depthltCal']
        out_sids = [f'{out_prefix}:image', f'{out_prefix}:world']
        recipe_sid = 'event:recipe:id'

        # optional output video
        if out_file is True:
            out_file = os.path.join(store_dir or self.STORE_DIR, replay or nowstring(), f'{self.output_prefix}.mp4')

        async with StreamReader(self.api, in_sids, [recipe_sid], recording_id=replay, fullspeed=fullspeed, raw_ts=True, **kw) as reader, \
                   StreamWriter(self.api, out_sids, test=test) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async def _stream():
                data = holoframe.load_all(self.api.data('depthltCal'))
                self.data.update(data)

                async for sid, t, x in reader:
                    # watch for recipe changes
                    if sid == recipe_sid:
                        self.change_recipe(x)
                        continue
                    
                    # update the central data store
                    self.data[sid] = x
                    self.ts[sid] = t

                    # compute machine learning
                    if sid == 'main':
                        im = x['image']
                        self.model_results = self.predict(im)
                        continue
                    elif self.model_results is None:
                        continue
                    
                    # get 3d transformer
                    main, depth, depthcal = [self.data[k] for k in in_sids]
                    mts, = [self.ts[k] for k in in_sids[:1]]
                    pts3d = self.get_pts3d(main, depth, depthcal)

                    # extract ml info
                    rgb = main['image']
                    h, w = rgb.shape[:2]
                    xyxy, confs, class_ids, labels = self.model_results
                    xywhn = xyxy.copy()
                    xywhn[:, 0] = (xywhn[:, 0]) / w
                    xywhn[:, 1] = (xywhn[:, 1]) / h
                    xywhn[:, 2] = (xywhn[:, 2] - xywhn[:, 0]) / w
                    xywhn[:, 3] = (xywhn[:, 3] - xywhn[:, 1]) / h

                    # get 3d results
                    xyz_top, xyz_center, dist = pts3d.transform_center_top(xyxy)
                    valid = dist < self.max_depth_dist  # make sure the points aren't too far

                    # write back to stream
                    await writer.write([
                        self.dump(
                            self.image_box_keys, 
                            [xywhn, confs, class_ids, labels]),
                        self.dump(
                            self.world_box_keys, 
                            [x[valid] for x in [xyz_center, xyz_top, confs, class_ids, labels]]),
                    ])

                    # optional video display
                    if imout.active:
                        imout.output(draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), xyxy, [
                            f'{l} {c:.0%} [{x:.0f},{y:.0f},{z:.0f}]' 
                            for l, c, (x,y,z) in zip(labels[valid], confs[valid], xyz_center[valid])
                        ]), parse_epoch_time(mts))

            # this lets us do two things at once
            #await asyncio.gather(_stream(), reader.watch_replay())
            await call_multiple_async(_stream(), reader.watch_replay())

    def change_recipe(self, recipe_id):
        if not recipe_id:
            self.model.set_vocab(DEFAULT_VOCAB)
            return
        recipe = self.api.recipes.get(recipe_id)
        self.model.set_vocab([w for k in ['tools_simple', 'ingredients_simple'] for w in recipe[k]])

    def predict(self, im):
        outputs = self.model(im)
        insts = outputs["instances"].to("cpu")
        xyxy = insts.pred_boxes.tensor.numpy()
        class_ids = insts.pred_classes.numpy().astype(int)
        confs = insts.scores.numpy()
        print(class_ids)
        labels = self.model.labels[class_ids]
        return xyxy, confs, class_ids, labels

    def get_pts3d(self, main, depthlt, depthltCal):
        return Points3D(
            main['image'], depthlt['image'], depthltCal['lut'],
            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']])

    def dump(self, keys, xs):
        return orjson.dumps([
            dict(zip(keys, xs)) for xs in zip(*xs)
        ], option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)

if __name__ == '__main__':
    import fire
    fire.Fire(Detic3D)
