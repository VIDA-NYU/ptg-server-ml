import collections
import pathlib

import itertools
DIR = pathlib.Path(__file__).parent
import os
import cv2
import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from .model_util import TensorQueue

import sys
sys.path.append("./omnivore")

has_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Omnivore(nn.Module):
    def __init__(self, vocab_subset=None, device=device):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # model
        self.device = device = torch.device(device)
        self.model = model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic")#.to(device)
        model.eval()

        self.heads = self.model.heads
        self.model.heads = nn.Identity()

        # Create an id to label name mapping
        self.data_dir = os.path.abspath(os.path.join(__file__, '../data'))
        import csv
        with open(os.path.abspath(os.path.join(self.data_dir, 'epic_action_classes.csv')), 'r') as f:
            labels = np.array([" ".join(rows) for rows in csv.reader(f)])
        self.full_labels = labels
        self.labels = labels
        self.label2index = {v.strip(): k for k, v in enumerate(self.full_labels)}
        if vocab_subset:
            self.set_vocab(vocab_subset)
        
        self._load_output_transform_matrix(self.label2index)
        # self.to(device)

    def _load_output_transform_matrix(self, label2index):
        verb_matrix, self.verb_idx, self.verb_labels = self._construct_matrix(
            label2index, os.path.join(self.data_dir, 'EPIC_100_verb_classes.csv'), 0)
        noun_matrix, self.noun_idx, self.noun_labels = self._construct_matrix(
            label2index, os.path.join(self.data_dir, 'EPIC_100_noun_classes.csv'), 1)
        self.register_buffer('verb_matrix', verb_matrix)
        self.register_buffer('noun_matrix', noun_matrix)

    def _construct_matrix(self, label2index, fname, i_a):
        import pandas as pd
        classes = pd.read_csv(fname, usecols=['id', 'key']).set_index('id').key
        idx = torch.zeros(len(label2index))
        matrix = torch.zeros(len(label2index), len(classes))
        for i, x in enumerate(classes):
            for a, j in label2index.items():
                if a.split(' ')[i_a] == x:
                    matrix[j,i] = 1.
                    idx[j] = i
        return matrix, idx, classes


    def set_vocab(self, vocab_subset):
        if not vocab_subset:
            self.labels = self.full_labels
            return

        self.idx_map = np.array([self.label2index[l] for l in vocab_subset])
        self.labels = np.array(vocab_subset)

    def prepare_image(self, im):
        # 1,C,H,W
        im = prepare_image(im, self.mean, self.std)
        # im = torch.as_tensor(im.transpose(2, 0, 1)[None], device=self.device)
        # im = FV.normalize(short_side_scale(im/255, 224), self.mean, self.std)
        return im

    def topk(self, y_pred, k=5):
        topk, i = torch.topk(y_pred, k=k)
        labels = self.labels[i[0]]
        return labels, topk[0]
    
    def forward(self, x, return_embedding=False):
        z = self.model(x, input_type="video")
        y = self.heads(z)
        if return_embedding:
            return y, z
        return y
    
    def project_verb_noun(self, y):
        return self.soft_project_verb_noun(y)
        # return self.project_verb_noun_single_top_k(y)

    def hard_project_verb_noun(self, y):
        # must relocate the following to be able to train
        # using these matrices will also make it impossible to
        # get topk accuracies for k>1
        verb_noun_index = torch.argmax(y, dim=-1, keepdims=True)
        y_hardmax = torch.zeros_like(y).scatter_(1, verb_noun_index, 1.0)
        verb = y_hardmax @ self.verb_matrix
        noun = y_hardmax @ self.noun_matrix
        return [verb, noun]
    
    def soft_project_verb_noun(self, y):
        y = F.softmax(y, dim=-1)
        verb = y @ self.verb_matrix
        noun = y @ self.noun_matrix
        # verb = F.softmax(verb, dim=-1)
        # noun = F.softmax(noun, dim=-1)
        # verb = torch.round(verb, decimals=2)
        return verb, noun
    
    def _proj_single_top_k(self, y_sort, label_idx, k=3):
        # get topk unique verb matches and their 
        matches = {}
        for v, yi in zip(label_idx, y_sort):
            if v in matches: continue
            matches[v] = yi
            if len(matches) > k:
                break
        labels, ys = zip(*sorted(matches.items(), key=lambda x: x[1], reverse=True))
        ys = F.softmax(torch.Tensor(ys), dim=-1)
        out = torch.zeros(len(label_idx))
        out[torch.tensor(labels)] = ys
        return out

    def project_verb_noun_single_top_k(self, y, k=3):
        sort_idx = torch.argsort(y, dim=-1)[::-1]
        y_sort = y[sort_idx]
        verb = self._proj_single_top_k(y_sort, self.verb_idx[sort_idx], k=k)
        noun = self._proj_single_top_k(y_sort, self.noun_idx[sort_idx], k=k)
        return verb, noun

# top n . out of, 
# 

def prepare_image(im, mean, std, expected_size=224):
    '''[H, W, 3] => [3, 224, 224]'''
    scale = max(expected_size/im.shape[0], expected_size/im.shape[1])
    im = cv2.resize(im, (0,0), fx=scale, fy=scale)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.astype(float) / 255
    im = (im - np.asarray(mean)) / np.asarray(std)
    im = crop_center(im, expected_size, expected_size)
    return torch.Tensor(im.transpose(2, 0, 1))

def short_side_scale(x, size):
    c, t, h, w = x.shape
    h, w = (int((float(h) / w) * size), size) if w < h else (size, int((float(w) / h) * size))
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

def crop_center(img,cropx,cropy):
    y,x,_ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

@torch.no_grad()
def run(src, out_file=None, fps=16, n_frames=32, stride=8, show=None):
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = Omnivore()
    

    if out_file is True:
        out_file='omnivore_'+os.path.basename(src)

    q = TensorQueue(n_frames, dim=1)
    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.push(model.prepare_image(im))

            if not i % stride and len(q) >= n_frames:
                y_pred = model.forward(q.tensor()[None]).cpu()
                # y_pred, noun_pred = model.project_verb_noun(y_pred)
                topk, y_top = model.topk(y_pred)
                _, idxs = torch.topk(y_pred, k=5)
                # topk = model.verb_labels[idxs[0].numpy()]
                topk = model.labels[idxs[0].numpy()]
                tqdm.tqdm.write(f'top: {" | ".join(topk)}')

            imout.output(draw_text_list(im, topk)[0])

if __name__ == '__main__':
    import fire
    fire.Fire(run)
