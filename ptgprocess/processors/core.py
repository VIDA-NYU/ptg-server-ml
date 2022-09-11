import asyncio
from ..util import maybe_profile



class Processor:
    def __init__(self, api=None):
        name = self.__class__.__name__
        if api is not False:
            import ptgctl
            self.api = api or ptgctl.API(username=name, password=name, should_log=False)

    async def call_async(self):
        raise NotImplementedError        

    @maybe_profile
    def run(self, *a, **kw):
        return asyncio.run(self.run_async(*a, **kw))

    async def run_async(self, *a, continuous=False, **kw):
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
                import traceback
                traceback.print_exc()
                #print(f'{type(e).__name__}: {e}')
