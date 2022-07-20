import os
import json
import torch
import clip
import asyncio
from PIL import Image
import redis.asyncio as aioredis
import ptgctl
from ptgctl import holoframe


device = "cuda" if torch.cuda.is_available() else "cpu"




# NOTE: the App base class is the generic code separated from the model specific code
class App:
    current_id = None

    def __init__(self, redis_url=None, id_key='recipe:id'):
        '''A Recipe-Dependent App. Will wait for an active recipe and then start inference.

        Arguments:
            redis_url (str): The url pointing to redis. Falls back to $REDIS_URL environment variable or default localhost url.
            id_key (str): The Redis key to use for checking redis.
        '''
        self.api = ptgctl.API(
            username=os.getenv('API_USER') or 'basic-clip', 
            password=os.getenv('API_PASS') or 'basic-clip')
        self.redis_url = redis_url or os.getenv('REDIS_URL') or 'redis://redis:6379'        
        self.id_key = id_key

    async def connect(self):
        '''Connect to redis.'''
        self.redis = await aioredis.from_url(self.redis_url)

    async def _get_id(self):
        '''Get the recipe ID from redis.'''
        rec_id = await self.redis.get(self.id_key)
        return rec_id.decode('utf-8') if rec_id else rec_id

    async def _wait_for_active(self, initial_id=None, delay=1):
        '''Wait for an active or changed recipe ID from redis.'''
        while True:
            self.current_id = rec_id = await self._get_id()
            if rec_id != initial_id:
                return rec_id
            await asyncio.sleep(delay)

    async def run_async(self, active_id=None, **kw):
        '''Run the main loop asynchronously.'''
        if active_id:  # force a recipe ID
            self.current_id = active_id
            return await self.run_while_active(active_id, **kw)
        while True:
            try:
                # print('waiting for recipe')
                active_id = await self._wait_for_active()
                await asyncio.gather(
                    self._wait_for_active(active_id),
                    self.run_while_active(active_id, **kw))
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def run_while_active(self, active_id, **kw):
        '''This is the main model code. This will run while a recipe is active.'''
        raise NotImplementedError

    async def read(self, sid, last, live=True, count=1, block=10000):
        '''Read the next item from redis.'''
        # if live:
        #     print(sid, '+', last, count, block, flush=True)
        #     d = await self.redis.xrevrange(sid, '+', last, count=count)
        #     if not d:
        #         return await self.redis.xread({sid: '$'}, count=count)
        #     return [sid, d]
        return await self.redis.xread({sid: last}, count=count, block=block)

    test = True
    async def upload(self, streams, ts='*', wrap_with_data_key=True):
        '''Upload results to redis'''
        if wrap_with_data_key:
            streams = {k: {b'd': d} for k, d in streams.items()}
        if self.test:
            print('-', ts, '------')
            for sid, data in streams.items():
                data = self._prepare_data(data)
                print(sid, data[b'd'])
            return
        async with self.redis.pipeline() as pipe:
            for sid, data in streams.items():
                data = self._prepare_data(data)
                pipe.xadd(sid, data, ts or '*')
            return await pipe.execute()

    def _prepare_data(self, data):
        '''Make sure both the keys and values are encoded as bytes.'''
        return {
            k.encode('utf-8') if not isinstance(k, bytes) else k: 
            json.dumps(d).encode('utf-8') if not isinstance(d, bytes) else d
            for k, d in data.items()
        }

    @classmethod
    @ptgctl.util.async2sync
    async def main(cls, recipe_id=None, last='$', **kw):
        '''Run the app.'''
        app = cls(**kw)
        await app.connect()
        if recipe_id:
            recipe_id = recipe_id or self.get_id()
            assert recipe_id
            await app.run_while_active(recipe_id)
        else:
            await app.run_async(recipe_id, last=last)


class ClipApp(App):
    tools_prompt = 'a photo of a {}'
    ingredients_prompt = 'a photo of a {}'
    instructions_prompt = '{}'

    def __init__(self, model_name="ViT-B/32", input_sid='main', out_prefix='clip:basic-zero-shot', **kw):
        '''A Basic Clip Zero-Shot Model
        
        Arguments:
            model_name (str): The Clip model name.
            input_sid (str): The camera stream name to use.
            out_prefix (str): The output stream prefix name. e.g. ``f'{out_prefix}:tools'``
            **kw: see ``App``.
        '''
        super().__init__(**kw)
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.input_sid = input_sid
        self.out_prefix = out_prefix

    async def run_while_active(self, recipe_id, last='$'):
        '''This is the main model code. 
        
        '''
        # load the recipe from the api
        recipe = self.api.recipes.get(recipe_id)
        # encode the tools, ingredients, and instructions using CLIP's text encoder.
        tools, z_tools = self.encode_text(recipe['tools'], self.tools_prompt)
        ingredients, z_ingredients = self.encode_text(recipe['ingredients'], self.ingredients_prompt)
        instructions, z_instructions = self.encode_text(recipe['instructions'], self.instructions_prompt)
        
        # sid is the camera stream name
        sid = self.input_sid
        # if live, it will always retrieve the latest frames. Otherwise, it will retrieve all frames.
        live = last == '$'
        print(last, live)
        # run this loop while the recipe is still the same (this is watched in App._wait_for_active)
        while recipe_id == self.current_id:
            # read the next image
            results = await self.read(sid, last, live=live)

            # iterate over the results
            for sid, samples in results:
                for ts, data in samples:
                    # update the latest timestamp
                    last = ts
                    # encode the image
                    z_image = self.encode_image(holoframe.load(data[b'd'])['image'])
                    # compare the image to text queries
                    tools_similarity = self.compare_image_text(z_image, z_tools)[0]
                    ingredients_similarity = self.compare_image_text(z_image, z_ingredients)[0]
                    instructions_similarity = self.compare_image_text(z_image, z_instructions)[0]

                    # upload results to the server
                    await self.upload({
                        f'{self.out_prefix}:tools': self._bundle(tools, tools_similarity),
                        f'{self.out_prefix}:ingredients': self._bundle(ingredients, ingredients_similarity),
                        f'{self.out_prefix}:instructions': self._bundle(instructions, instructions_similarity),
                    }, ts)
    
    def encode_text(self, texts, prompt_format=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        if prompt_format:
            texts = [prompt_format.format(x) for x in texts]
        toks = clip.tokenize(texts).to(device)
        z = self.model.encode_text(toks)
        z /= z.norm(dim=-1, keepdim=True)
        return texts, z

    def encode_image(self, image):
        '''Encode image to CLIP embedding.'''
        image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(device)
        z_image = self.model.encode_image(image)
        z_image /= z_image.norm(dim=-1, keepdim=True)
        return z_image

    def compare_image_text(self, z_image, z_text):
        '''Compare image and text similarity (not sure why the 100, it's from the CLIP repo).'''
        return (100.0 * z_image @ z_text.T).softmax(dim=-1)

    def _bundle(self, text, similarity):
        '''Prepare text and similarity to be uploaded.'''
        return dict(zip(text, similarity.tolist()))


import collections
class OnlineStats:
    '''Computing online statistics. - i.e. do online mean pooling step-by-step.
    
    Parameters:
        mean (float): The rolling mean.
        var (float): The rolling variance.
        std (float): The rolling standard deviation.
        last (float): The last value.
    '''
    mean = _var = 0
    def __init__(self, n=3):
        self.samples = collections.deque(maxlen=n)
        self.win_size = n

    def clear(self):
        self.samples.clear()
        self.mean = self._var = 0

    def append(self, x):
        '''Append a value to the rolling mean.'''
        c = len(self.samples)
        drop = c >= self.win_size
        drop_x = self.samples[0] if drop else 0
        last_mean = self.mean
        self.mean = m = (last_mean * c - drop_x + x) / (c + int(not drop))
        if c:
            self._var += (x - last_mean) * (x - m)
        self.last = x
        self.samples.append(x)

    @property
    def sum(self):
        '''Get the rolling sum.'''
        return self.mean * self.count

    @property
    def std(self):
        '''Get the rolling standard deviation.'''
        return self.var ** 0.5

    @property
    def var(self):
        '''Get the rolling variance.'''
        c = len(self.samples)
        return self._var / (c - 1) if c > 1 else 0



if __name__ == '__main__':
    import fire
    fire.Fire(ClipApp.main)
