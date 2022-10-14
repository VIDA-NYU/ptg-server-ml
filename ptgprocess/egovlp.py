from __future__ import annotations
import os
import sys
import collections
import tqdm

os.environ['LOCAL_RANK'] = os.getenv('LOCAL_RANK') or '0'

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

MODEL_DIR = os.getenv('MODEL_DIR') or 'models' # localfile('../models')

EGOVLP_CHECKPOINT = os.path.join(MODEL_DIR, 'epic_mir_plus.pth')

# EGOVLP_DIR=os.getenv('EGOVLP_DIR') or 'EgoVLP'
EGOVLP_DIR = os.path.join(os.path.dirname(__file__), 'EgoVLP')
sys.path.append(EGOVLP_DIR)


# https://github.com/showlab/EgoVLP/blob/main/configs/eval/epic.json
class EgoVLP(nn.Module):
    norm_mean=(0.485, 0.456, 0.406)
    norm_std=(0.229, 0.224, 0.225)
    def __init__(self, checkpoint=EGOVLP_CHECKPOINT, input_res=224, center_crop=256, n_samples=16, device=device):  #  tokenizer_model="distilbert-base-uncased"
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
        self.device = device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
        # image transforms
        self.transforms = T.Compose([
            T.Resize(center_crop),
            T.CenterCrop(center_crop),
            T.Resize(input_res),
            NormalizeVideo(mean=self.norm_mean, std=self.norm_std),
        ])

    def forward(self, video, text, return_sim=True):
        with torch.no_grad():
            text_embed, vid_embed = self.model({'video': video, 'text': text}, return_embeds=True)
            if return_sim:
                return self.similarity(text_embed, vid_embed)
            return vid_embed, text_embed

    def encode_text(self, text, prompt=None):
        '''Encode text prompts. Returns formatted prompts and encoded CLIP text embeddings.'''
        with torch.no_grad():
            if self.tokenizer is not None:
                if prompt:
                    text = [prompt.format(t) for t in text]
                text = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            return self.model.compute_text({key: val.to(self.device) for key, val in text.items()})

    def encode_video(self, video):
        with torch.no_grad():
            return self.model.compute_video(video)

    def prepare_image(self, im):
        im = im[:,:,::-1]
        im = im.transpose(2, 0, 1)
        im = torch.as_tensor(im.astype(np.float32))[:,None] / 255
        im = self.transforms(im)
        im = im.transpose(1, 0)
        return im

    def add_image(self, im):
        self.q.append(self.prepare_image(im))
        return self

    def predict_recent(self):
        return self.encode_video(torch.stack(list(self.q), dim=1).to(self.device))

    def similarity(self, z_text, z_video, dual=False):
        return similarity(z_text, z_video, dual=dual)


def similarity(z_text, z_video, temp=100, temp_inv=1/500, dual=False):
    if dual:
        sim = sim_matrix_mm(z_text, z_video)
        sim = F.softmax(sim*temp_inv, dim=1) * sim
        sim = F.softmax(temp*sim, dim=0)
        return sim.t()
    sim = (sim_matrix(z_text, z_video) + 1) / 2
    return F.softmax(temp*sim.t(), dim=1)


def sim_matrix(a, b, eps=1e-8):
    #added eps for numerical stability
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    return sim_matrix_mm(a_norm, b_norm)

def sim_matrix_mm(a, b):
    return torch.mm(a, b.transpose(0, 1))


def run(src, vocab, include=None, exclude=None, out_file=None, n_frames=16, fps=10, stride=1, show=None, ann_root=None, **kw):
    from ptgprocess.record import CsvWriter
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list, get_vocab
    model = EgoVLP(**kw)

    if out_file is True:
        out_file='egovlp_'+os.path.basename(src)

    vocab = get_vocab(vocab, ann_root, include, exclude)
    q = collections.deque(maxlen=n_frames)

    z_text = model.encode_text(vocab)
    print(z_text.shape)

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout, \
         CsvWriter('egovlp_'+os.path.basename(src), header=['_time']+list(vocab)) as csvout:
        topk = topk_text = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.append(model.prepare_image(im))
            print(len(q))
            if not i % stride:
                z_video = model.encode_video(torch.stack(list(q), dim=1).to(device))
                sim = similarity(z_text, z_video, dual=True).detach()[0]
                i_topk = torch.topk(sim, k=5).indices.tolist()
                topk = [vocab[i] for i in i_topk]
                topk_text = [f'{vocab[i]} ({sim[i]:.2%})' for i in i_topk]

                tqdm.tqdm.write(f'top: {topk}')
            imout.output(draw_text_list(im, topk_text)[0])
            
            csvout.write([t] + sim.tolist())

if __name__ == '__main__':
    import fire
    fire.Fire(run)
