"""
Author Jianzhe Lin
May.2, 2020
"""
import cv2
import matplotlib.pyplot as plt
import os
import re_id
import orjson
import logging
import numpy as np
import ptgctl
import ptgctl.holoframe
import ptgctl.util
from collections import defaultdict


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

class MemoryApp:
    def __init__(self):
        self.api = ptgctl.API(username=os.getenv('API_USER') or 'memory',
                              password=os.getenv('API_PASS') or 'memory')
        self.re_id = re_id.ReId()

    @ptgctl.util.async2sync
    async def run(self, prefix=None):
        prefix = prefix or ''
        input_sid = f'{prefix}detic:world'
        output_sid = f'{prefix}detic:memory'

        async with self.api.data_pull_connect(input_sid) as ws_pull, \
                   self.api.data_push_connect(output_sid, batch=True) as ws_push:
            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    objects = orjson.loads(data)

                    for obj in objects:
                        label, seen_before = self.re_id.update_memory(np.asarray(obj['xyz_center']), obj['labels'])
                        obj['track_id'] = label
                        obj['seen_before'] = seen_before

                    await ws_push.send_data([orjson.dumps(objects)])


if __name__ == '__main__':
    import fire
    fire.Fire(MemoryApp)
