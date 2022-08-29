from __future__ import annotations
import os
import asyncio
import orjson
import tqdm

import numpy as np

import ptgctl
from ptgctl import holoframe
from ptgctl.util import parse_epoch_time
from .core import Processor
from .util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_boxes



class Yolo3D(Processor):
    output_prefix = 'yolo3d'
    image_box_keys = ['xyxy', 'confidence', 'class_id']
    world_box_keys = ['xyz_tl', 'xyz_br', 'xyz_tr', 'xyz_bl', 'xyzc', 'confidence', 'class_id']
    min_dist_secs = 1
    STORE_DIR = 'post'

    def __init__(self):
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        labels = self.model.names
        if isinstance(labels, dict):
            labels = {int(k): l for k, l in self.model.names.items()}
            labels = [labels.get(i, i) for i in range(max(labels))]
        self.labels = np.asarray(labels)

        super().__init__()

    async def call_async(self, replay=None, fullspeed=None, prefix=None, out_file=None, fps=10, store_dir=None, **kw):
        import cv2
        from ptgctl.pt3d import Points3D
        self.data = {}
        self.last_timestamp = None

        prefix = prefix or ''
        out_prefix = f'{prefix}{self.output_prefix}'
        in_sids = ['main', 'depthlt', 'depthltCal']
        out_sids = [f'{out_prefix}:image', f'{out_prefix}:world']
        if out_file is True:
            out_file = os.path.join(store_dir or self.STORE_DIR, replay or nowstring(), f'{self.output_prefix}.mp4')

        async with StreamReader(self.api, in_sids, recording_id=replay, fullspeed=fullspeed, merged=True) as reader, \
                   StreamWriter(self.api, out_sids, test=True) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=True) as imout:
            async def _stream():
                data = holoframe.load_all(self.api.data(f'{replay if replay else ""}:depthltCal'))
                if replay:
                    data = {k[len(replay)+1:]:v for k, v in data.items()}
                self.data.update(data)

                async for data in reader:
                    self.data.update(data)
                    try:
                        main, depthlt, depthltCal = [self.data[k] for k in in_sids]
                        rgb = main['image']

                        # check time difference
                        mts = main['timestamp']
                        dts = depthlt['timestamp']
                        secs = parse_epoch_time(mts) - parse_epoch_time(dts)
                        if abs(secs) > self.min_dist_secs:
                            raise KeyError(f"timestamps too far apart main={mts} depth={dts} ∆{secs:.3g}s")
                        if mts == self.last_timestamp:
                            continue
                        self.last_timestamp = mts
                        
                        # create point transformer
                        pts3d = Points3D(
                            rgb, depthlt['image'], depthltCal['lut'],
                            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
                            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']])
                    except KeyError as e:
                        tqdm.tqdm.write(f'KeyError: {e} {set(self.data)}')
                        continue

                    xyxy = self.model(rgb).xyxy[0].numpy()
                    confs = xyxy[:, 4]
                    class_ids = xyxy[:, 5].astype(int)
                    labels = self.labels[class_ids]
                    xyxy = xyxy[:, :4]
                    xyz_tl, xyz_br, xyz_tr, xyz_bl, xyzc, dist = pts3d.transform_box(xyxy[:, :4])
                    valid = dist < 5  # make sure the points aren't too far

                    # await writer.write([
                    #     self.dump(self.image_box_keys, [x[valid] for x in [xyxy, confs, class_ids, labels]]),
                    #     self.dump(self.world_box_keys, [x[valid] for x in [xyz_tl, xyz_br, xyz_tr, xyz_bl, xyzc, confs, class_ids, labels]]),
                    # ])
                    imout.output(draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), xyxy, [
                        f'{l} {c:.0%} [{x:.0f},{y:.0f},{z:.0f}]' 
                        for l, c, (x,y,z) in zip(labels[valid], confs[valid], xyzc[valid])
                    ]), parse_epoch_time(mts))

            await asyncio.gather(_stream(), reader.watch_replay())

    def dump(self, keys, xs):
        return orjson.dumps([
            dict(zip(keys, xs)) for xs in zip(*xs)
        ], option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)

if __name__ == '__main__':
    import fire
    fire.Fire({
        '3d': Yolo3D,
    })