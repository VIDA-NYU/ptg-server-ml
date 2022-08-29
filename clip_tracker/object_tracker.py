from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

import torch
from torch import nn

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from ptgprocess.util import video_feed, ImageOutput, draw_boxes


class Yolo(nn.Module):
    def __init__(self):
        super().__init__()




def nms(boxes, scores, threshold):
    n_active = len(boxes)
    active = np.array([True]*len(boxes))
    idxs = np.argsort(scores)

    for i, ii in enumerate(idxs):
        if not n_active: break
        if not active[i]: continue
        for j, jj in enumerate(idxs[i+1:], i+1):
            if not active[j]: continue
            if IoU(boxes[ii], boxes[jj]) > threshold:
                active[j] = False
                n_active -= 1
    return idxs[active]

def IoU(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    areaA = aw * ah
    areaB = bw * bh
    if areaA <= 0 or areaB <= 0: return 0
    intersectionArea = (
        max(min(ay + ah, by + bh) - max(ay, by), 0) * 
        max(min(ax + aw, bx + bw) - max(ax, bx), 0))
    return intersectionArea / (areaA + areaB - intersectionArea)


def image_loader(im, imsize=640):
    im = cv2.resize(im, (imsize,imsize))#int(imsize*im.shape[1]/im.shape[0])
    img = im[:, :, ::-1].transpose(2, 0, 1) 
    img = torch.from_numpy(np.ascontiguousarray(img)).float()
    img /= 255.0
    return img[None], im

def run(src, min_confidence=0.8, nms_max_overlap=0.3, min_height=0, max_cosine_distance=0.2, nn_budget=None, 
        out_file=None, fps=10, show=None):
    """Run multi-target tracker on a particular sequence.
    
    Arguments:
        min_confidence (float): Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
        nms_max_overlap (float): Maximum detection overlap (non-maxima suppression threshold).
        min_height (int): Detection height threshold. Disregard all detections that have a height lower than this value.
        max_cosine_distance (float): Gating threshold for cosine distance metric (object appearance).
        nn_budget (int): Maximum size of the appearance descriptor gallery. If None, no budget is enforced.
    """
    tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.eval()

    if out_file is True:
        out_file='tracked_'+os.path.basename(src)

    with ImageOutput(out_file, fps, show=show) as imout:
        for i, im in video_feed(src):
            X_im,  im = image_loader(im)
            xywh = model(X_im)[0]
            xywh = xywh[xywh[:,4] >= min_confidence]
            xywh = xywh.clone()
            xywh[:,[2,3]] -= xywh[:,[0,1]]
            xywh = xywh[(xywh[:,3] >= min_height)]
            indices = nms(xywh[:, :4], xywh[:, 4], nms_max_overlap)
            feature = None
            print(xywh)

            # Update tracker.
            tracker.predict()
            tracker.update([
                Detection(xywh[i, :4], xywh[i, 4], xywh[5:])
                for i in indices
            ])

            tracks = [
                track for track in tracker.tracks 
                if not track.is_confirmed() or track.time_since_update > 0
            ]

            # imout.output(im)
            imout.output(draw_boxes(
                im, 
                [d.to_tlwh() for d in tracks], 
                [d.track_id for d in tracks]))


if __name__ == '__main__':
    import fire
    fire.Fire(run)
