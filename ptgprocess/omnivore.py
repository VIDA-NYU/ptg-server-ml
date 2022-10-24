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

import sys
sys.path.append("./omnivore")

has_gpu = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Omnivore(nn.Module):
    def __init__(self, n_frames=10, vocab_subset=None, device=device):
        super().__init__()
        self.device = device = torch.device(device)
        self.model = model = torch.hub.load("facebookresearch/omnivore:main", model="omnivore_swinB_epic").to(device)
        model.eval()

        self.n_frames = n_frames

        self.q = collections.deque(maxlen=self.n_frames)

        # Create an id to label name mapping
        import csv
        with open(os.path.abspath(os.path.join(__file__, '../data/epic_action_classes.csv')), 'r') as f:
            labels = [" ".join(rows) for rows in csv.reader(f)]
        self.mask = None
        if vocab_subset:
            self.idx_map = np.array([labels.index(l) for l in vocab_subset])
            labels = vocab_subset
        # print('vocab', self.video_labels)
        self.labels = np.array(labels)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def prepare_image(self, im):
        # 1,C,H,W
        im = torch.as_tensor(im.transpose(2, 0, 1)[None], device=self.device)
        im = FV.normalize(short_side_scale(im/255, 224), self.mean, self.std)
        # im_crops = [uniform_crop(im, 224, i)[0] for i in [0,1,2]]
        return im[0]

    def add_image(self, im):
        self.q.append(self.prepare_image(im))

    def predict_recent(self):
        # T,C,H,W
        ims = torch.stack(list(itertools.islice(itertools.cycle(self.q), self.n_frames)), dim=1)
        return self.predict(ims[None], 'video')

    def forward(self, im):
        self.add_image(im)
        return self.predict_recent()

    def predict(self, input, input_type='video'):
        # The model expects inputs of shape: B x C x T x H x W
        with torch.no_grad():
            return self.model(input.to(self.device), input_type=input_type)

    def topk(self, y_pred, k=5):
        topk, i = torch.topk(y_pred, k=k)
        labels = self.labels[i[0]]
        return labels, topk[0]


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


def run(src, n_frames=30, out_file=None, fps=10, stride=1, show=None):
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = Omnivore(n_frames=n_frames)

    if out_file is True:
        out_file='omnivore_'+os.path.basename(src)

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            model.add_image(im)
            if not i % stride:
                y_pred = model.predict_recent()
                topk, y_top = model.topk(y_pred)
                tqdm.tqdm.write(f'top: {topk}')
            imout.output(draw_text_list(im, topk)[0])

if __name__ == '__main__':
    import fire
    fire.Fire(run)
