import os
import orjson
import logging
import numpy as np
import ptgctl
from ptgctl import holoframe
import ptgctl.util
from ptgctl import pt3d

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

class MemoryApp:
    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'pointcloud',
                              password=os.getenv('API_PASS') or 'pointcloud')

    def get_pts3d(self, main, depthlt, depthltCal):
        return Points3D(
            main['image'], depthlt['image'], depthltCal['lut'],
            depthlt['rig2world'], depthltCal['rig2cam'], main['cam2world'],
            [main['focalX'], main['focalY']], [main['principalX'], main['principalY']])

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        prefix = prefix or ''
        input_sid = f'{prefix}detic:world'
        output_sid = f'{prefix}detic:memory'

        data = {}

        in_sids = ['main', 'depthlt', 'depthltCal']

        async with self.api.data_pull_connect(input_sid) as ws_pull, \
                   self.api.data_push_connect(output_sid, batch=True) as ws_push:
            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    self.data[sid].update(holoframe.load(data))
                    if sid == 'main':
                        self = holoframe.load(data)['image']
                        continue
                    if sid == 'depthlt':
                        self.depthlt




if __name__ == '__main__':
    import fire
    fire.Fire(MemoryApp)
