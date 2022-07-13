import os
import time
import asyncio
import orjson
import numpy as np
import ptgctl
import ptgctl.holoframe
from ptgctl.pt3d import Points3D

import warnings
warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'")


def unpack_entries(offsets: list, content: bytes) -> list:
    '''Unpack a single bytearray with numeric offsets into multiple byte objects.'''
    print(offsets)
    entries = []
    for (sid, ts, i), (_, _, j) in zip(offsets, offsets[1:] + [(None, None, None)]):
        entries.append((sid, ts, content[i:j]))
    return entries
ptgctl.util.unpack_entries = unpack_entries

class App:
    def __init__(self, min_dist_secs=1):
        self.min_dist_secs = min_dist_secs
        self.api = ptgctl.CLI(
            username=os.getenv('API_USER') or 'yolo', 
            password=os.getenv('API_PASS') or 'yolo')
        self.load_model()
        self.calibrate()

    def load_model(self):
        import torch
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
        self.model.amp = False
        self.labels = np.asarray(self.model.names)

    def calibrate(self):
        data = self.api.data('depthltCal')
        #print(data)
        data = ptgctl.holoframe.load_all(data)
        #print(set(data), flush=True)
        self.lut, self.T_rig2cam = ptgctl.holoframe.unpack(data, [
            'depthltCal.lut', 
            'depthltCal.rig2cam', 
        ])

    def run(self, **kw):
        while True:
            try:
                return asyncio.run(self.run_async(**kw))
            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(3)
                continue

    async def run_async(self, **kw):
        streams = ['main', 'depthlt']
        async with self.api.data_pull_connect('+'.join(streams), time_sync_id='main', **kw) as wsr:
            # async with self.api.data_push_connect('+'.join(streams), **kw) as wsw:
                while True:
                    data = await wsr.recv_data()
                    try:
                        if not data:
                            print('empty data')
                            await asyncio.sleep(0.1)
                            continue
                        data = ptgctl.holoframe.load_all(data)
                        print(set(data), flush=True)
                        ts = ptgctl.util.parse_time(data['main']['timestamp'])
                        print('timestamp difference:', {
                            k: str(ts - ptgctl.util.parse_time(d['timestamp']))
                            for k, d in data.items() if 'timestamp' in d
                        })
                        self.data.update(data)
                    except Exception:
                        await report_error("problem retrieving data", 1)
                        continue

                    try:
                        try:
                            pts3d, rgb = self.get_pts3d()
                        except KeyError as e:
                            print("missing data:", e)
                            continue
                        
                        results = self.process_data(rgb, pts3d)
                        output = orjson.dumps(self.as_json(results), option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
                        # wsw.send_data(output)
                        print(output)

                    except Exception:
                        await report_error("problem processing data", 0.1)
                        continue

    def get_pts3d(self):
        mts, dts = ptgctl.holoframe.unpack('main.timestamp', 'depthlt.timestamp')
        secs = parse_epoch_time(mts) - parse_epoch_time(dts)
        if abs(secs) > self.min_dist_secs:
            raise KeyError(f"timestamps too far apart main={mts} depth={dts} ∆{secs:.3g}s")

        (
            rgb, depth,
            T_rig2world, T_pv2world,
            focalX, focalY, principalX, principalY,
        ) = ptgctl.holoframe.unpack(
            self.data, [
            'main.image',
            'depthlt.image',
            'depthlt.rig2world',
            'main.cam2world',
            'main.focalX',
            'main.focalY',
            'main.principalX',
            'main.principalY',
        ])

        pts3d = Points3D(
            rgb, depth, self.lut,
            T_rig2world, self.T_rig2cam, T_pv2world,
            [focalX, focalY],
            [principalX, principalY])
        return pts3d, rgb

    def process_data(self, rgb, pts3d):
        results = self.model(rgb)
        xyxy = results.xyxy[0].numpy()
        meta = xyxy[:, 4:]

        (
            xyz_tl_world, xyz_br_world, 
            xyz_tr_world, xyz_bl_world, 
            xyzc_world, dist,
        ) = pts3d.transform_box(xyxy[:, :4])
        valid = dist < 5  # make sure the points aren't too far

        print(xyxy.shape, xyz_tl_world.shape)
        print(valid.sum(), dist)
        print(xyzc_world)

        xs = [xyz_tl_world, xyz_br_world, xyz_tr_world, xyz_bl_world, xyzc_world, meta]
        xs = [x[valid] for x in xs]
        return np.concatenate(xs, axis=1)

    columns = [
        'x_tl', 'y_tl', 'z_tl', 
        'x_br', 'y_br', 'z_br', 
        'x_tr', 'y_tr', 'z_tr', 
        'x_bl', 'y_bl', 'z_bl', 
        'xc', 'yc', 'zc', 
        'confidence', 'class_id']
    def as_json(self, results):
        return [
            dict(zip(self.columns, d), label=self.labels[int(d[-1])])
            for d in results
        ]




import traceback

async def report_error(msg, sleep=0.1):
    print(msg)
    traceback.print_exc()
    await asyncio.sleep(sleep)


if __name__ == '__main__':
    import fire
    fire.Fire(App)

