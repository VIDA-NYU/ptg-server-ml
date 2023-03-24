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
    def __init__(self, n_frames=10, vocab_subset=None, device=device):
        super().__init__()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # model
        self.n_frames = n_frames
        self.device = device = torch.device(device)
        self.model = model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic").to(device)
        model.eval()

        # self.q = collections.deque(maxlen=self.n_frames)

        # Create an id to label name mapping
        self.data_dir = os.path.join(__file__, '../data')
        import csv
        with open(os.path.abspath(os.path.join(self.data_dir, 'epic_action_classes.csv')), 'r') as f:
            labels = np.array([" ".join(rows) for rows in csv.reader(f)])
        self.full_labels = labels
        self.labels = labels
        self.label2index = {v: k for k, v in enumerate(self.full_labels)}
        if vocab_subset:
            self.set_vocab(vocab_subset)
        
        verb_matrix, noun_matrix = self._get_output_transform_matrix(self.label2index)
        self.register_buffer('verb_matrix', verb_matrix)
        self.register_buffer('noun_matrix', noun_matrix)

    def set_vocab(self, vocab_subset):
        if not vocab_subset:
            self.labels = self.full_labels
            return

        self.idx_map = np.array([self.label2index[l] for l in vocab_subset])
        self.labels = np.array(vocab_subset)

    def prepare_image(self, im):
        # 1,C,H,W
        im = torch.as_tensor(im.transpose(2, 0, 1)[None], device=self.device)
        im = FV.normalize(short_side_scale(im/255, 224), self.mean, self.std)
        # im_crops = [uniform_crop(im, 224, i)[0] for i in [0,1,2]]
        return im[0]

    # def add_image(self, im):
    #     self.q.append(self.prepare_image(im))

    # def predict_recent(self):
    #     # T,C,H,W
    #     ims = torch.stack(list(itertools.islice(itertools.cycle(self.q), self.n_frames)), dim=1)
    #     return self.predict(ims[None], 'video')

    # def forward(self, im):
    #     self.add_image(im)
    #     return self.predict_recent()

    def predict(self, input, input_type='video'):
        # The model expects inputs of shape: B x C x T x H x W
        with torch.no_grad():
            return self.model(input.to(self.device), input_type=input_type)

    def topk(self, y_pred, k=5):
        topk, i = torch.topk(y_pred, k=k)
        labels = self.labels[i[0]]
        return labels, topk[0]
    

    def _get_output_transform_matrix(self, label2index):
        verb_matrix = self._construct_matrix(label2index, os.path.join(self.data_dir, 'EPIC_100_verb_classes.csv'), 0)
        noun_matrix = self._construct_matrix(label2index, os.path.join(self.data_dir, 'EPIC_100_noun_classes.csv'), 1)
        return verb_matrix, noun_matrix

    def _construct_matrix(self, label2index, fname, i_a):
        import pandas as pd
        classes = pd.read_csv(fname, usecols=['id', 'key']).set_index('id').key
        matrix = torch.zeros(len(label2index), len(classes))
        for i, x in enumerate(classes):
            for a, j in label2index.items():
                if a.split(',')[i_a] == x:
                    matrix[j,i] = 1.
        return matrix

    def forward(self, x):
        y = self.model(x, input_type="video")

    def project_verb_noun(self, y):
        # must relocate the following to be able to train
        # using these matrices will also make it impossible to
        # get topk accuracies for k>1
        verb_noun_index = torch.argmax(y, dim=-1, keepdims=True)
        y_hardmax = torch.zeros_like(y).scatter_(1, verb_noun_index, 1.0)
        verb = y_hardmax @ self.verb_matrix
        noun = y_hardmax @ self.noun_matrix
        return [verb, noun]


def short_side_scale(x, size):
    c, t, h, w = x.shape
    h, w = (int((float(h) / w) * size), size) if w < h else (size, int((float(w) / h) * size))
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

def uniform_crop(images, size, spatial_idx, boxes=None, scale_size=None):
    assert spatial_idx in [0, 1, 2]
    h, w = images.shape[2:4]
    if scale_size is not None:
        short_side_scale(images, scale_size)
    y_offset = int(np.ceil((h - size) / 2))
    x_offset = int(np.ceil((w - size) / 2))
    if h > w:
        y_offset = h - size if spatial_idx == 2 else 0 if spatial_idx == 0 else y_offset
    else:
        x_offset = w - size if spatial_idx == 2 else 0 if spatial_idx == 0 else x_offset
    return images[:, :, y_offset:y_offset + size, x_offset:x_offset + size]


def run(src, n_frames=16, out_file=None, fps=10, stride=1, show=None):
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = Omnivore(n_frames=n_frames)

    if out_file is True:
        out_file='omnivore_'+os.path.basename(src)

    q = TensorQueue(n_frames)
    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.push(model.prepare_image(im))

            if not i % stride:
                y_pred = model.predict(q.tensor())
                topk, y_top = model.topk(y_pred)
                tqdm.tqdm.write(f'top: {topk}')

            imout.output(draw_text_list(im, topk)[0])

if __name__ == '__main__':
    import fire
    fire.Fire(run)
