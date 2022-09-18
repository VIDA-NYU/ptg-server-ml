"""
Author Jianzhe Lin
May.2, 2020
"""
import os
import re_id
import orjson
import logging
import numpy as np
import ptgctl
import ptgctl.holoframe
import ptgctl.util


logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

#ptgctl.log.setLevel('WARNING')

RECIPE_SID = 'event:recipe:id'

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

        async with self.api.data_pull_connect([input_sid, RECIPE_SID]) as ws_pull, \
                   self.api.data_push_connect(output_sid, batch=True) as ws_push:
            while True:
                for sid, timestamp, data in await ws_pull.recv_data():
                    # clear the memory when a new recipe is started
                    if sid == RECIPE_SID:
                        self.re_id = re_id.ReId()
                        continue
                    
                    objects = orjson.loads(data)
                    self.re_id.update_frame(objects)
                    await ws_push.send_data([orjson.dumps(self.re_id.dump_memory())])


if __name__ == '__main__':
    import fire
    fire.Fire(MemoryApp)
