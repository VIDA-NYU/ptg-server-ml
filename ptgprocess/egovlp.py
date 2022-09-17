from __future__ import annotations
import os
import sys
import collections
import tqdm

import cv2
import numpy as np
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms._transforms_video import NormalizeVideo
import transformers


device = "cuda" if torch.cuda.is_available() else "cpu"
localfile = lambda *fs: os.path.abspath(os.path.join(os.path.dirname(__file__), *fs))

MODEL_DIR = os.getenv('MODEL_DIR') or localfile('../models')

EGOVLP_CHECKPOINT = os.path.join(MODEL_DIR, 'epic_mir_plus.pth')

EGOVLP_DIR=os.getenv('EGOVLP_DIR') or 'EgoVLP'
sys.path.append(EGOVLP_DIR)


# https://github.com/showlab/EgoVLP/blob/main/configs/eval/epic.json
class EgoVLP(nn.Module):
    norm_mean=(0.485, 0.456, 0.406)
    norm_std=(0.229, 0.224, 0.225)
    def __init__(self, checkpoint=EGOVLP_CHECKPOINT, input_res=224, center_crop=256, n_samples=10, device=device, **kw):  #  tokenizer_model="distilbert-base-uncased"
        super().__init__()
        self.q = collections.deque(maxlen=n_samples)
        print(checkpoint)
        
        from model.model import FrozenInTime
        model = FrozenInTime(**{
            "video_params": {
                "model": "SpaceTimeTransformer",
                "arch_config": "base_patch16_224",
                "num_frames": 16,
                "pretrained": True,
                "time_init": "zeros"
            },
            "text_params": {
                "model": "distilbert-base-uncased",
                "pretrained": True,
                "input": "text"
            },
            "projection": "minimal",
            "load_checkpoint": checkpoint or None,
        })
        ## load model
        #checkpoint = torch.load(checkpoint)
        #state_dict = checkpoint['state_dict']
        ##state_dict = state_dict_data_parallel_fix(state_dict, model.state_dict())
        #model.load_state_dict(state_dict, strict=True)
        self.model = model.to(device)
        self.model.eval()
        self.device = self.model.device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
        # image transforms
        self.transforms = T.Compose([
            T.Resize(center_crop),
            T.CenterCrop(center_crop),
            T.Resize(input_res),
            NormalizeVideo(mean=self.norm_mean, std=self.norm_std),
        ])

    def forward(self, video, text):
        text_embed, vid_embed = self.model({}, return_embeds=True)

    def encode_text(self, text, prompt=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        with torch.no_grad():
            if self.tokenizer is not None:
                if prompt:
                    text = [prompt.format(t) for t in text]
                text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            return self.model.compute_text({key: val.to(self.device) for key, val in text.items()})

    def add_image(self, im):
        im = im[:,:,::-1]
        im = im.transpose(2, 0, 1)
        im = torch.as_tensor(im.astype(np.float32))[:,None] / 255
        im = self.transforms(im)
        im = im.transpose(1, 0)
        self.q.append(im)
        return self

    def predict_recent(self):
        X = torch.stack(list(self.q), dim=1).to(self.device)
        z_video = self.model.compute_video(X)
        return z_video

    def similarity(self, z_text, z_video, dual=False):
        if dual:
            sim = sim_matrix_mm(z_text, z_video)
            sim = F.softmax(sim / 500, dim=1) * sim
            sim = F.softmax(100*sim, dim=0)
            return sim.t()
        sim = (sim_matrix(z_text, z_video) + 1) / 2
        return F.softmax(100*sim.t(), dim=1)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return sim_matrix_mm(a_norm, b_norm)

def sim_matrix_mm(a, b):
    return torch.mm(a, b.transpose(0, 1))


# def sim_matrix(a, b, eps=1e-8):
#     """added eps for numerical stability"""
#     a_n = F.normalize(a, dim=1, eps=eps)
#     b_n = F.normalize(b, dim=1, eps=eps)
#     sim_mt = torch.mm(a, b.transpose(0, 1))
#     return sim_mt
# 
# def sim_matrix_mm(a, b):
#     sim_mt = torch.mm(a, b.transpose(0, 1))
#     return sim_mt









def get_vocab(vocab, ann_root):
    if vocab is None:
        return vocab  # someone elses problem lol
    if isinstance(vocab, (list, tuple)):
        return vocab  # literal
    if ':' in vocab:
        kind, vocab, key = vocab.split(':', 2)
        kind = kind.lower()
        if kind == 'recipe':
            import ptgctl
            api = ptgctl.API()
            recipe = api.recipes.get(vocab)
            return [w for k in key.split(',') for w in recipe[k]]
        if kind.startswith('ek'):
            import pandas as pd
            df = pd.concat([
                pd.read_csv(os.path.join(ann_root, "EPIC_100_train_normalized.csv")).assign(split='train'),
                pd.read_csv(os.path.join(ann_root, "EPIC_100_validation_normalized.csv")).assign(split='val'),
            ])
            df = df[df.video_id == vocab] if vocab != 'all' else df
            if key not in df.columns:
                raise ValueError(f'{key} not in {df.columns}')
            return df[key].unique().tolist()
    raise ValueError("Invalid vocab")




def run(src, vocab, vocab_key='steps_simple', out_file=None, n_frames=20, fps=10, stride=1, show=None, ann_root=None, **kw):
    from ptgprocess.record import CsvWriter
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = EgoVLP(n_frames=n_frames, **kw)


    if out_file is True:
        out_file='egovlp_'+os.path.basename(src)

    vocab = get_vocab(vocab, ann_root)

    z_text = model.encode_text(vocab)
    print(z_text.shape)

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout, \
         CsvWriter('egovlp_'+os.path.basename(src), header=['_time']+list(vocab)) as csvout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            model.add_image(im)
            if not i % stride:
                z_video = model.predict_recent()
                sim = model.similarity(z_text, z_video, dual=False).detach()[0]
                i_topk = torch.topk(sim, k=5).indices.tolist()
                topk = [vocab[i] for i in i_topk]
                topk_text = [f'{vocab[i]} ({sim[i]:.2%})' for i in i_topk]

                tqdm.tqdm.write(f'top: {topk}')
            imout.output(draw_text_list(im, topk_text)[0])
            
            csvout.write([t] + sim.tolist())

if __name__ == '__main__':
    import fire
    fire.Fire(run)
