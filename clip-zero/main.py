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


class App:    
    tools_prompt = 'a photo of a {}'
    ingredients_prompt = 'a photo of a {}'
    instructions_prompt = '{}'

    current_id = None

    def __init__(self, model_name="ViT-B/32", url=None, 
                 id_key='recipe:id', 
                 input_stream='main', 
                 out_prefix='clip:basic-zero-shot'):
        self.api = ptgctl.API(
            username=os.getenv('API_USER') or 'basic-clip', 
            password=os.getenv('API_PASS') or 'basic-clip')
        self.redis_url = url or os.getenv('REDIS_URL') or 'redis://localhost:6379'
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.id_key = id_key
        self.input_stream = input_stream
        self.out_prefix = out_prefix

    async def connect(self):
        self.redis = await aioredis.from_url(self.redis_url)

    async def get_id(self):
        rec_id = await self.redis.get(self.id_key)
        return rec_id.decode('utf-8') if rec_id else rec_id

    async def _wait_for_active_recipe_id(self, initial_id=None, delay=1):
        while True:
            self.current_id = rec_id = await self.get_id()
            if rec_id != initial_id:
                return rec_id
            await asyncio.sleep(delay)

    async def run(self):
        while True:
            try:
                print('waiting for recipe')
                rec_id = await self._wait_for_active_recipe_id()
                await asyncio.gather(
                    self._wait_for_active_recipe_id(rec_id),
                    self.run_recipe(rec_id))
            except Exception:
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def run_recipe(self, recipe_id, last='$'):
        recipe = self.api.recipes.get(recipe_id)
        tools, z_tools = self.encode_text(recipe['tools'], self.tools_prompt)
        ingredients, z_ingredients = self.encode_text(recipe['ingredients'], self.ingredients_prompt)
        instructions, z_instructions = self.encode_text(recipe['instructions'], self.instructions_prompt)

        sid = self.input_stream
        live = last == '$'
        while recipe_id == self.current_id:
            results = await self.read(sid, last, live=live)
            print('read', len(results), last, live)
            for sid, samples in results:
                for ts, data in samples:
                    last = ts
                    z_image = self.encode_image(holoframe.load(data[b'd'])['image'])
                    tools_similarity = self.compare_image_text(z_image, z_tools)[0]
                    ingredients_similarity = self.compare_image_text(z_image, z_ingredients)[0]
                    instructions_similarity = self.compare_image_text(z_image, z_instructions)[0]

                    await self.upload({
                        f'{self.out_prefix}:tools': self.make_stream_data(tools, tools_similarity),
                        f'{self.out_prefix}:ingredients': self.make_stream_data(ingredients, ingredients_similarity),
                        f'{self.out_prefix}:instructions': self.make_stream_data(instructions, instructions_similarity),
                    }, ts)

    async def read(self, sid, last, live=True, count=1):
        if live:
            return [sid, await self.redis.xrevrange(sid, '$', last, count=count)]
        return await self.redis.xread({sid: last}, count=count)

    def encode_text(self, texts, prompt_format=None):
        if prompt_format:
            texts = [prompt_format.format(x) for x in texts]
        toks = clip.tokenize(texts).to(device)
        z = self.model.encode_text(toks)
        z /= z.norm(dim=-1, keepdim=True)
        return texts, z

    def encode_image(self, image):
        image = Image.fromarray(image)
        image = self.preprocess(image).unsqueeze(0).to(device)
        z_image = self.model.encode_image(image)
        z_image /= z_image.norm(dim=-1, keepdim=True)
        return z_image

    def compare_image_text(self, z_image, z_text):
        return (100.0 * z_image @ z_text.T).softmax(dim=-1)

    def make_stream_data(self, keys, values):
        data = dict(zip(keys, values.tolist()))
        return {b'd': json.dumps(data)}

    async def upload(self, streams, ts='*', test=True):
        if test:
            print('-', ts, '------')
            for sid, data in streams.items():
                print(sid, data[b'd'])
            return
        async with self.redis.pipeline() as pipe:
            for sid, data in streams.items():
                pipe.xadd(sid, data, ts or '*')
            return await pipe.execute()

    def list_recipes(self):
        return self.api.recipes.ls()

    async def start_recipe(self, rec_id):
        assert rec_id
        await self.redis.set(self.id_key, rec_id)
        return rec_id

    async def stop_recipe(self):
        return await self.redis.delete(self.id_key)


def async_wrap(func):
    import functools
    @functools.wraps(func)
    def inner(*a, **kw):
        return asyncio.run(func(*a, **kw))
    inner.asyncio = func
    return inner


@async_wrap
async def run(**kw):
    app = App(**kw)
    await app.connect()
    await app.run()

@async_wrap
async def run_recipe(recipe_id, last=0, **kw):
    app = App(**kw)
    await app.connect()
    app.current_id = recipe_id
    await app.run_recipe(recipe_id, last=last)

@async_wrap
async def start(recipe_id, **kw):
    app = App(**kw)
    await app.connect()
    print(await app.start_recipe(recipe_id))

@async_wrap
async def stop(**kw):
    app = App(**kw)
    await app.connect()
    print(await app.stop_recipe())


if __name__ == '__main__':
    funcs = [run, start, stop, run_recipe]
    import fire
    fire.Fire({f.__name__: f for f in funcs})
