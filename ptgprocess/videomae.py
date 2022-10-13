from __future__ import annotations
import os
import sys
import collections
import tqdm
import numpy as np

import cv2
import torch
from torch import nn
import transformers
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification


device = "cuda" if torch.cuda.is_available() else "cpu"


class VideoMAE(nn.Module):
    def __init__(self, device=device):
        super().__init__()
        self.fe = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-short-finetuned-ssv2")
        self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-short-finetuned-ssv2")
        self.model.to(device)
        self.device = device
        self.labels = self.model.config.id2label
        if isinstance(self.labels, dict):
            self.labels = np.array([self.labels.get(i, '') for i in range(max(self.labels)+1)])

    def forward(self, video):
        with torch.no_grad():
            encoding = self.fe(video, return_tensors="pt")
            #print(encoding.pixel_values.shape)
            outputs = self.model(encoding.pixel_values.to(device))
            return outputs.logits
              


def run(src, out_file=None, n_frames=16, fps=10, stride=1, show=None, ann_root=None, **kw):
    from ptgprocess.record import CsvWriter
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list
    model = VideoMAE(**kw)

    if out_file is True:
        out_file='videomae_'+os.path.basename(src)

    q = collections.deque(maxlen=n_frames)

    vocab = model.labels

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout, \
         CsvWriter('videomae_'+os.path.basename(src), header=['_time']+list(vocab)) as csvout:
        
        topk = topk_text = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            #print(im.shape)
            q.append(im)
            if len(q) < n_frames:
                continue

            if not i % stride:
                logits = model(list(q))[0]
                i_topk = torch.topk(logits, k=5).indices.tolist()
                topk = [vocab[i] for i in i_topk]
                topk_text = [f'{vocab[i]} ({logits[i]:.2%})' for i in i_topk]

                tqdm.tqdm.write(f'top: {topk}')
                csvout.write([t] + logits.tolist())
            imout.output(draw_text_list(im, topk_text)[0])
            

if __name__ == '__main__':
    import fire
    fire.Fire(run)
