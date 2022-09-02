from __future__ import annotations
import os
import collections

import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import clip


device = "cuda" if torch.cuda.is_available() else "cpu"
localfile = lambda *fs: os.path.join(os.path.dirname(__file__), *fs)
ACTION1_CHECKPOINT = localfile('../models/epoch=2-step=99021.ckpt')
ACTION2_CHECKPOINT = localfile('../models/model_best.pt')

class ZeroClip(nn.Module):
    def __init__(self, model_name="ViT-B/32", **kw):
        super().__init__()
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.tokenize = clip.tokenize

    def encode_text(self, texts, prompt_format=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        toks = self.tokenize([prompt_format.format(x) for x in texts] if prompt_format else texts).to(device)
        z = self.model.encode_text(toks)
        z = F.normalize(z, dim=1)
        return z

    def encode_image(self, im):
        '''Encode image to CLIP embedding.'''
        im = self.preprocess(Image.fromarray(im[...,::-1]))[None].to(device)
        z_image = self.model.encode_image(im)
        z_image = F.normalize(z_image, dim=-1)
        return z_image

    def compare_image_text(self, z_image, z_text):
        '''Compare image and text similarity (not sure why the 100, it's from the CLIP repo).'''
        return (100 * z_image @ z_text.T).softmax(dim=-1)



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

    def integrate_time(self, z_image):
        self.q_z_im.append(z_image)
        z_image = torch.stack(list(self.q_z_im))
        z_image = z_image.mean(dim=0)
        z_image = F.normalize(z_image, dim=-1)
        return z_image

    def encode_image(self, im):
        z_image = super().encode_image(im)
        z_image = self.integrate_time(z_image)
        return z_image



class ActionClip2(ZeroClip):
    def __init__(self, n_samples=10, checkpoint=None, **kw):
        super().__init__(**kw)
        checkpoint = torch.load(
            checkpoint or os.path.join(os.path.dirname(__file__), '../models/model_best.pt'),
            map_location=torch.device(device))

        wrapper = nn.Module()
        self.fusion = wrapper.module = visual_prompt(checkpoint['model_state_dict'], n_samples).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        wrapper.load_state_dict(checkpoint['fusion_model_state_dict'])
        self.q_z_im = collections.deque(maxlen=n_samples)

    def clear(self):
        self.q_z_im.clear()

    def integrate_time(self, z_image):
        self.q_z_im.append(z_image)
        z_image = torch.stack(list(self.q_z_im), dim=1)
        z_image = self.fusion(z_image)
        z_image = F.normalize(z_image, dim=-1)
        return z_image

    def encode_image(self, im):
        z_image = super().encode_image(im)
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



MODELS = {
    'zero': ZeroClip,
    'action1': ActionClip1,
    'action2': ActionClip2,
}