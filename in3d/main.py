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
from ptgctl import pt3d

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

#ptgctl.log.setLevel('WARNING')


class dicque(OrderedDict):
    def __init__(self, *a, maxlen=0, **kw):
        self._max = maxlen
        super().__init__(*a, **kw)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)

    def closest(self, tms):
        log.debug(str([abs(tms-k) for k in self]))
        k = min(self, key=lambda t: abs(tms-t), default=None)
        return k

MAX_TIME_DIFF_BETWEEN_MAIN_AND_DEPTH = 300

class In3DApp:
    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'pointcloud',
                              password=os.getenv('API_PASS') or 'pointcloud')
        self.max_depth_dist = 7

    def get_pts3d(self, main, depthlt, depthltCal):
        return pt3d.Points3D(
            main['image'], depthlt['image'], depthltCal['lut'],
            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']],
            generate_point_cloud=True)
    
    def get_detic3d(self, detic, depthlt, depthltCal):
        return pt3d.Points3D(
            detic['shape'][:2], depthlt['image'], depthltCal['lut'],
            depthlt['rig2world'], depthltCal['rig2cam'], detic['cam2world'],
            detic['focal'], detic['principal'],
            generate_point_cloud=False)

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        prefix = prefix or ''
        input_sid = f'{prefix}detic:world'
        output_sid = f'{prefix}detic:memory'

        data = {}
        hist = dicque(maxlen=20)

        in_sids = ['main', 'depthlt', 'depthltCal']
        obj_sid = 'detic:image:for3d'
        out_sids = ['pointcloud:v2', 'detic:world']
        skipped_labels = {'person'}

        pbar = tqdm.tqdm()
        async with self.api.data_pull_connect(in_sids + [obj_sid], ack=True) as ws_pull, \
                   self.api.data_push_connect(out_sids, batch=True) as ws_push:
            with logging_redirect_tqdm():
            
                data.update(holoframe.load_all(self.api.data('depthltCal')))
                while True:
                    for sid, t, buffer in await ws_pull.recv_data():
                        pbar.update()
                        tms = int(t.split('-')[0])

                        # project detic
                        if sid == obj_sid:
                            k = hist.closest(tms)
                            if k is None or abs(k-tms) > MAX_TIME_DIFF_BETWEEN_MAIN_AND_DEPTH:
                                log.debug('no pt3d for detic t=%d', tms)
                                continue
                            log.debug('using detic t=%d, im=%d dt=%d', tms, k, tms-k)
                            
                            d = orjson.loads(buffer)
                            pt3d = self.get_detic3d(d["image_params"], hist[k], data['depthltCal'])
                            d = d["objects"]
                            h,w = pt3d.im_shape
                            
                            xyxy = np.array([o['xyxyn'] for o in d]).reshape(-1, 4)
                            xyxy[:, 0] *= w 
                            xyxy[:, 1] *= h 
                            xyxy[:, 2] *= w 
                            xyxy[:, 3] *= h 
                            xyz_center, dist = pt3d.transform_center_withinbbox(xyxy)
                            valid = dist < self.max_depth_dist  # make sure the points aren't too far
                            log.debug('%d/%d boxes valid. dist in [%f,%f]', valid.sum(), len(valid), dist.min(initial=np.inf), dist.max(initial=0))

                            for obj, xyz_center, valid, dist in zip(d, xyz_center, valid, dist):
                                obj['xyz_center'] = xyz_center
                                obj['depth_map_dist'] = dist
                            
                            d = list(filter(lambda x: x['label'] not in skipped_labels, d))
                            await ws_push.send_data([jsondump(d)], ['detic:world'], [t])
                            continue

                        # store the data
                        data[sid] = d = holoframe.load(buffer)
                        d['_tms'] = tms
                        d['_t'] = t
                        
                        if sid != 'depthlt':
                            continue
    
                        # get pt3d
                        try:
                            main, depthlt, depthltCal = [data[k] for k in in_sids]
                            mts = main['_tms']
                            dts = depthlt['_tms']
                            if abs(mts-dts) > MAX_TIME_DIFF_BETWEEN_MAIN_AND_DEPTH:
                                continue
                            pts3d = self.get_pts3d(main, depthlt, depthltCal)
                        except KeyError as e:
                            print(e)
                            await asyncio.sleep(0.1)
                            continue


                        log.debug('created pt3d main=%d depth=%d dt=%d', mts, dts, dts-mts)
                        
                        # save for later
                        hist[depthlt['_tms']] = depthlt

                        # write point cloud
                        # await ws_push.send_data([
                        #     jsondump({
                        #         'time': dts,
                        #         'color': pts3d.rgb,
                        #         'xyz_world': pts3d.xyz_depth_world,
                        #     }),
                        # ], ['pointcloud'], [dts])

                        dts = depthlt['_t']
                        x, y, z = pts3d.xyz_depth_world.T
                        r, g, b = pts3d.rgb.T
                        await ws_push.send_data([
                            pqdump({
                                'time': [int(dts.split('-')[0])]*len(x),
                                "x": x,
                                "y": y,
                                "z": z,
                                "r": r,
                                "g": g,
                                "b": b,
                            }),
                        ], ['pointcloudv2'], [dts])
                        

import pyarrow as pa
import pyarrow.parquet as pq

def pqdump(data):
    table = pa.table(data)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf)
    return buf.getvalue().to_pybytes()

def jsondump(data):
    return orjson.dumps(data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)




if __name__ == '__main__':
    import fire
    fire.Fire(In3DApp)
