from __future__ import annotations
import os
import asyncio
import orjson
import tqdm
import collections

import numpy as np
from PIL import Image
import cv2

import torch
from torch import nn
import torch.nn.functional as F
import clip

import ptgctl
from ptgctl import holoframe
from ptgctl.util import parse_epoch_time
from .core import Processor
from .util import StreamReader, StreamWriter, ImageOutput, nowstring, draw_text_list


device = "cuda" if torch.cuda.is_available() else "cpu"
localfile = lambda *fs: os.path.join(os.path.dirname(__file__), *fs)
ACTION1_CHECKPOINT = localfile('../models/epoch=2-step=99021.ckpt')
ACTION2_CHECKPOINT = localfile('../models/model_best.pt')

class ZeroClip(nn.Module):
    def __init__(self, model_name="ViT-B/32", **kw):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.tokenize = clip.tokenize

    def encode_text(self, texts, prompt_format=None, return_texts=False):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        toks = self.tokenize([prompt_format.format(x) for x in texts] if prompt_format else texts).to(device)
        z = self.model.encode_text(toks)
        z /= z.norm(dim=-1, keepdim=True)
        return (z, texts) if return_texts else z

    def encode_image(self, image):
        '''Encode image to CLIP embedding.'''
        image = self.preprocess(Image.fromarray(image[...,::-1]))[None].to(device)
        z_image = self.model.encode_image(image)
        z_image /= z_image.norm(dim=-1, keepdim=True)
        return z_image

    def compare_image_text(self, z_image, z_text):
        '''Compare image and text similarity (not sure why the 100, it's from the CLIP repo).'''
        return (100 * (z_image @ z_text.T)).softmax(dim=-1)



class ActionClip1(ZeroClip):
    def __init__(self, n_samples=10, checkpoint=ACTION1_CHECKPOINT, **kw):
        super().__init__(**kw)
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=device)['state_dict']
            wrapper = nn.Module()
            wrapper.model = self.model
            wrapper.load_state_dict(collections.OrderedDict({
                k: v for k, v in state_dict.items()
                if not k.startswith('teacher.') and k not in {'sink_temp'}
            }))
        self.q_z_im = collections.deque(maxlen=n_samples)

    def clear(self):
        self.q_z_im.clear()

    def encode_single_image(self, im):
        im = self.preprocess(Image.fromarray(im[:,:,::-1]))[None].to(device)
        return self.model.encode_image(im)

    def integrate_time(self, z_image):
        self.q_z_im.append(z_image)
        z_image = torch.stack(list(self.q_z_im))
        z_image = z_image.mean(dim=0, keepdim=True)
        z_image = F.normalize(z_image, dim=1)
        return z_image

    def encode_image(self, im):
        z_image = self.encode_single_image(im)
        z_image = self.integrate_time(z_image)
        return z_image



class ActionClip2(ZeroClip):
    def _load_model(self, n_samples=10, checkpoint=None, **kw):
        super()._load_model(**kw)
        checkpoint = torch.load(
            checkpoint or os.path.join(os.path.dirname(__file__), '../models/model_best.pt'),
            map_location=torch.device(device))

        wrapper = nn.Module()
        self.fusion = wrapper.module = visual_prompt(checkpoint['model_state_dict'], n_samples)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper.load_state_dict(checkpoint['fusion_model_state_dict'])
        self.q_z_im = collections.deque(maxlen=n_samples)

    def clear(self):
        self.q_z_im.clear()

    def encode_single_image(self, im):
        im = self.preprocess(Image.fromarray(im[:,:,::-1]))[None].to(device)
        return self.model.encode_image(im)

    def integrate_time(self, z_image):
        self.q_z_im.append(z_image)
        z_image = torch.stack(list(self.q_z_im))
        print(z_image.shape)
        z_image = self.fusion(z_image)
        print(z_image.shape)
        z_image = z_image.mean(dim=0, keepdim=True)
        print(z_image.shape)
        z_image = F.normalize(z_image, dim=1)
        print(z_image.shape)
        input()
        return z_image

    def encode_image(self, im):
        z_image = self.encode_single_image(im)
        z_image = self.integrate_time(z_image)
        return z_image



class visual_prompt(nn.Module):
    def __init__(self, clip_state_dict, T=5):
        super().__init__()
        self.T = T

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        # vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        # transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x_original = x = x.contiguous()
        position_ids = torch.arange(t, dtype=torch.long, device=x.device)[None].expand(x.size(0), -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original
        return x.mean(dim=1, keepdim=False)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x












class ZeroClipProcessor(Processor):
    output_prefix = 'zero_clip'
    prompts = {
        # 'tools': 'a photo of a {}',
        # 'ingredients': 'a photo of a {}',
        'steps': '{}',
    }
    STORE_DIR = 'post'

    Model = ZeroClip

    async def call_async(self, recipe_id=None, *a, **kw):
        # if recipe_id:
        return await self._call_async(recipe_id, *a, **kw)
    #     recipe_id = self.api.sessions.current_recipe()
    #     if not recipe_id:
    #         print("waiting for recipe to be activated")
    #         recipe_id = await self._watch_recipe_id(recipe_id)
    #     return await asyncio.gather(
    #         self._call_async(recipe_id, *a, **kw),
    #         self._watch_recipe_id(recipe_id)
    #     )

    # async def _watch_recipe_id(self, recipe_id):
    #     loop = asyncio.get_event_loop()
    #     while True:
    #         new_recipe_id, _ = await asyncio.gather(
    #             loop.run_in_executor(None, self.api.sessions.current_recipe),
    #             asyncio.sleep(3)
    #         )
    #         if new_recipe_id != recipe_id:
    #             return new_recipe_id

    async def _call_async(self, recipe_id, replay=None, fullspeed=None, out_file=None, fps=5, show=True, store_dir=None, **kw):
        if not hasattr(self, 'model'):
            self.model = self.Model(**kw)
        self.current_id = recipe_id
        assert recipe_id, "You must provide a recipe ID, otherwise we have nothing to compare"
        # load the recipe from the api
        recipe = self.api.recipes.get(recipe_id)
        texts = {k: recipe[k] for k, _ in self.prompts.items()}
        z_texts = {k: self.model.encode_text(recipe[k], prompt) for k, prompt in self.prompts.items()}

        if out_file is True:
            out_file = os.path.join(
                store_dir or self.STORE_DIR, replay or nowstring(), 
                f'{self.__class__.__name__}-{self.output_prefix}.mp4')

        out_keys = set(texts)
        out_sids = [f'{replay or ""}{self.output_prefix}:{k}' for k in out_keys]
        async with StreamReader(self.api, ['main'], recording_id=replay, fullspeed=fullspeed) as reader, \
                   StreamWriter(self.api, out_sids, test=True) as writer, \
                   ImageOutput(out_file, fps, fixed_fps=True, show=show) as imout:
            async def _stream():
                async for _, t, d in reader:
                    if recipe_id and self.current_id != recipe_id:
                        print("recipe changed", recipe_id, '->', self.current_id)
                        break
                    # encode the image and compare to text queries
                    im = d['image']
                    z_image = self.model.encode_image(im)
                    if z_image is None:
                        continue
                    sims = {k: self.model.compare_image_text(z_image, z_texts[k])[0] for k in out_keys}
                    # await writer.write([self._bundle(texts[k], sims[k]) for k in out_keys])
                    imout.output(draw_text_list(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [
                        f'{texts[k][i]} ({sims[k][i]:.0%})' 
                        for k in out_keys 
                        for i in torch.topk(sims[k], 3, dim=-1)[1].tolist()
                    ])[0], t)

            await asyncio.gather(_stream(), reader.watch_replay())

    def _bundle(self, text, similarity):
        '''Prepare text and similarity to be uploaded.'''
        return dict(zip(text, similarity.tolist()))



class ActionClip1Processor(ZeroClipProcessor):
    Model = ActionClip1


class ActionClip2Processor(ZeroClipProcessor):
    Model = ActionClip2



if __name__ == '__main__':
    import fire
    fire.Fire({
        'zero': ZeroClipProcessor,
        'action1': ActionClip1Processor,
        'action2': ActionClip2Processor,
    })