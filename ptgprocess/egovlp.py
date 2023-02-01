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

    def similarity(self, z_text, z_video, **kw):
        return similarity(z_text, z_video, **kw)

    def few_shot_predictor(self, vocab, vocab_dir='fewshot'):
        vocab_dir = os.path.join(vocab_dir, vocab_dir)
        if os.path.isdir(vocab_dir):
            return FewShotPredictor(vocab_dir)

    def zero_shot_predictor(self, vocab):
        return ZeroShotPredictor(vocab, self)

    def get_predictor(self, vocab, texts=None, vocab_dir='fewshot'):
        pred = None
        if isinstance(vocab, str):
            pred = self.few_shot_predictor(vocab, vocab_dir)
        if pred is None:
            if texts is None and vocab is not None:
                texts = vocab
            if callable(texts):
                texts = texts()
            pred = self.zero_shot_predictor(vocab)
        return pred


def similarity(z_text, z_video, temp=1, temp_inv=1/500, dual=False):
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





class ZeroShotPredictor(nn.Module):
    def __init__(self, vocab, model, prompt='{}'):
        super().__init__()
        self.model = model
        self.vocab = np.asarray(vocab)
        tqdm.tqdm.write(f'zeroshot with: {vocab}')
        self.Z_vocab = self.model.encode_text(vocab, prompt)

    def forward(self, Z_image):
        '''Returns the action probabilities for that frame.'''
        scores = self.model.similarity(self.Z_vocab, Z_image).detach()
        return scores

class FewShotPredictor(nn.Module):
    def __init__(self,  vocab_dir, n_neighbors=33):
        super().__init__()

        # load all the data
        assert os.path.isdir(vocab_dir)
        fsx = sorted(glob.glob(os.path.join(vocab_dir, 'X_*.npy')))
        fsy = sorted(glob.glob(os.path.join(vocab_dir, 'Y_*.npy')))
        assert all(
            os.path.basename(fx).split('_', 1)[1] == os.path.basename(fy).split('_', 1)[1]
            for fx, fy in zip(fsx, fsy)
        )
        fvocab = os.path.join(vocab_dir, 'classes.npy')

        # load and convert to one big numpy array
        X = np.concatenate([np.load(f) for f in fsx])
        Y = np.concatenate([np.load(f) for f in fsy])
        self.vocab = vocab = np.asarray(np.load(fvocab))
        tqdm.tqdm.write(f'loaded {X.shape} {Y.shape}. {len(vocab)} {vocab}')

        # train the classifier
        tqdm.tqdm.write('training classifier...')
        #self.clsf = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.clsf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor')
        self.clsf.fit(X, Y)
        print(self.clsf.classes_)
        tqdm.tqdm.write('trained!')

    def forward(self, Z_image):
        scores = self.clsf.predict_proba(Z_image.cpu().numpy())
        return scores

def l2norm(Z, eps=1e-8):
    return Z / np.maximum(np.linalg.norm(Z, keepdims=True), eps)

def normalize(Z, eps=1e-8):
    Zn = Z.norm(dim=1)[:, None]
    return Z / torch.max(Zn, eps*torch.ones_like(Zn))






class FrameInput:
    def __init__(self, src, src_fps, fps, give_time=True, fallback_previous=True):
        self.src = src
        self.fps = fps or src_fps
        self.src_fps = src_fps
        self.give_time = give_time
        self.fallback_previous = fallback_previous

    def fname2i(self, f):
        return int(os.path.splitext(os.path.basename(f))[0].split('_')[-1])

    @staticmethod
    def cvt_fps(src_fps, fps):
        return int(max(round(src_fps / (fps or src_fps)), 1))

    def __enter__(self): return self
    def __exit__(self, *a): pass
    def __iter__(self):
        import cv2
        fs = os.listdir(os.path.dirname(self.src))
        i_max = self.fname2i(max(fs))
        every = self.cvt_fps(self.src_fps, self.fps)
        print(f'{self.src}: fps {self.src_fps} to {self.fps}. taking every {every} frames')

        im = None
        for i in tqdm.tqdm(range(0, i_max+1, every)):
            t = i / self.src_fps if self.give_time else i

            f = self.src.format(i)
            if not os.path.isfile(f):
                tqdm.tqdm.write(f'missing frame: {f}')
                if self.fallback_previous and im is not None:
                    yield t, im
                continue

            im = cv2.imread(f)
            yield t, im



def run(src, vocab, include=None, exclude=None, out_file=None, n_frames=16, fps=8, stride=1, show=None, ann_root=None, **kw):
    from ptgprocess.record import CsvWriter
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list, get_vocab

    model = EgoVLP(**kw)

    if out_file is True:
        out_file='egovlp_'+os.path.basename(src) + ('.mp4' if os.path.isdir(src) else '')

    vocab = get_vocab(vocab, ann_root, include, exclude)
    q = collections.deque(maxlen=n_frames)

    emissions = []
    Z_videos = []

    z_text = model.encode_text(vocab)
    print(z_text.shape)

    
    with FrameInput(src, 30, fps) if os.path.isdir(src) else VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout, \
         CsvWriter('egovlp_'+os.path.basename(src), header=['_time']+list(vocab)) as csvout:
        topk = topk_text = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.append(model.prepare_image(im))
            if not i % stride:
                z_video = model.encode_video(torch.stack(list(q), dim=1).to(device))
                sim = similarity(z_text, z_video, dual=False).detach()[0]
                i_topk = torch.topk(sim, k=5).indices.tolist()
                topk = [vocab[i] for i in i_topk]
                topk_text = [f'{vocab[i]} ({sim[i]:.2%})' for i in i_topk]

                tqdm.tqdm.write(f'top: {topk}')
                
                emissions.append(sim.cpu().numpy())
                Z_videos.append(z_video.cpu().numpy())

            imout.output(draw_text_list(im, topk_text)[0])
            
            csvout.write([t] + sim.tolist())

    if out_file:
        emissions = np.asarray(emissions).T
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 8), dpi=300)
        plt.imshow(np.asarray(emissions), aspect='auto', cmap='magma', origin='lower', interpolation='nearest')
        plt.yticks(range(len(vocab)), vocab)
        plt.colorbar()
        plt.savefig(os.path.splitext(out_file)[0]+'_emissions.png')
        np.savez(os.path.splitext(out_file)[0]+'_emissions.npz', emissions=emissions, vocab=vocab)



def extract(src, out_dir, file_pattern='frame_{:010d}.png', fps=8, src_fps=30, n_frames=16):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(src.strip(os.sep)))[0]
    name = f'{name}-n_frames={n_frames}-fps={fps}.npz'
    out_file = os.path.join(out_dir, name)
    print(out_file)
    #input()

    model = EgoVLP()

    q = collections.deque(maxlen=n_frames)
    
    frames = []
    Zs = []
    with (FrameInput(os.path.join(src, file_pattern), src_fps, fps, give_time=False) if os.path.isdir(src) else VideoInput(src, fps, give_time=False)) as vin:
        for i, (ii, im) in enumerate(vin):
            q.append(model.prepare_image(im))
            z_video = model.encode_video(torch.stack(list(q), dim=1).to(device))
            Zs.append(z_video.cpu().numpy())
            frames.append(ii)
    name = os.path.splitext(os.path.basename(src))[0]
    name = f'{name}-n_frames={n_frames}-fps={fps}.npz'
    np.savez(out_file, Z_video=Zs, frame=frames)


if __name__ == '__main__':
    import fire
    fire.Fire()
