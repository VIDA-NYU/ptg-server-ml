import collections
import pathlib

import itertools
DIR = pathlib.Path(__file__).parent
import os
import cv2
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
            self.video_labels = [" ".join(rows) for rows in csv.reader(f)]
        self.mask = None
        if vocab_subset:
            vocab_subset = set(vocab_subset)
            self.mask = np.ones(len(self.video_labels))
            self.mask[[l in vocab_subset for l in self.video_labels]] = 0
        # print('vocab', self.video_labels)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def forward(self, im, k=5):
        # 1,C,H,W
        im = torch.as_tensor(im.transpose(2, 0, 1)[None], device=self.device)
        im = FV.normalize(short_side_scale(im/255, 224), self.mean, self.std)
        # im_crops = [uniform_crop(im, 224, i)[0] for i in [0,1,2]]
        self.q.append(im[0])
        # T,C,H,W
        ims = itertools.islice(itertools.cycle(self.q), self.n_frames)
        ims = torch.stack(list(ims), dim=1)
        return self._predict_top_k(ims[None], 'video', self.video_labels, k=k)

    def _predict_top_k(self, input, input_type, cls_map, k=5):
        # The model expects inputs of shape: B x C x T x H x W
        with torch.no_grad():
            prediction = self.model(input.to(self.device), input_type=input_type)
            if self.mask is not None:
                prediction[self.mask] = -torch.inf
            pred_classes = prediction.topk(k=k).indices
        return [cls_map[i] for i in pred_classes[0]]


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


def run(src, out_file=None, fps=10, stride=1, show=None):
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = Omnivore()

    if out_file is True:
        out_file='omnivore_'+os.path.basename(src)

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            if not i % stride:
                topk = model(im)
                print(topk)
            imout.output(draw_text_list(im, topk)[0])

if __name__ == '__main__':
    import fire
    fire.Fire(run)
