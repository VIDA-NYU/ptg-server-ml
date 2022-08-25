import asyncio
import inspect
import ptgctl
from ptgctl.util import async2sync
from .util import maybe_profile



class Processor:
    def __init__(self, api=None):
        name = self.__class__.__name__
        self.api = api or ptgctl.API(username=name, password=name)

    async def call_async(self):
        raise NotImplementedError        

    @maybe_profile
    @async2sync
    async def __call__(self, *a, continuous=False, **kw):
        while True:
            try:
                await self.call_async(*a, **kw)
                if not continuous:
                    return
            except KeyboardInterrupt:
                print('byee!')
                return
            except Exception as e:
                if not continuous:
                    raise
                print(f'{type(e).__name__}: {e}')
