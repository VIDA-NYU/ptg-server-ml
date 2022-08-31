import os
import tqdm

import cv2
import numpy as np
import torch
from torch import nn

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from detectron2.utils.visualizer import Visualizer, random_color, ColorMode

from ptgprocess.detic import Detic
from ptgprocess.util import video_feed, ImageOutput, draw_boxes


# class Yolo(nn.Module):
#     def __init__(self, min_confidence=0.4, nms_max_overlap=0.6, min_height=0):
#         super().__init__()
#         self.model = model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#         model.eval()
#         # self.min_confidence = min_confidence
#         # self.nms_max_overlap = nms_max_overlap
#         # self.min_height = min_height
    
#     def forward(self, x):
#         xywh = self.model(x).xywh[0].numpy()
#         # print(xywh.shape, xywh[:3,:4])
#         # max_conf = xywh[:,4].max()
#         # xywh = xywh[xywh[:,4] >= self.min_confidence]

#         # xywh[:, 0] = xywh[:, 0] / x.shape[1]
#         # xywh[:, 1] = xywh[:, 1] / x.shape[0]
#         # xywh[:, 2] = (xywh[:, 2] - xywh[:, 0]) / x.shape[1]
#         # xywh[:, 3] = (xywh[:, 3] - xywh[:, 1]) / x.shape[0]
#         # xywh[:,[2,3]] -= xywh[:,[0,1]]
#         # xywh = xywh[(xywh[:,3] >= self.min_height)]
#         # indices = nms(xywh[:, :4], xywh[:, 4], self.nms_max_overlap)
#         # xywh = xywh[indices]
#         # tqdm.tqdm.write(f'{xywh.shape} max confidence: {max_conf}')
#         return xywh[:,:4], xywh[:,4], xywh[:,5:]


# class Detic(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


# def nms(boxes, scores, threshold):
#     n_active = len(boxes)
#     active = np.array([True]*len(boxes))
#     idxs = np.argsort(scores)

#     for i, ii in enumerate(idxs):
#         if not n_active: break
#         if not active[i]: continue
#         for j, jj in enumerate(idxs[i+1:], i+1):
#             if not active[j]: continue
#             if IoU(boxes[ii], boxes[jj]) > threshold:
#                 active[j] = False
#                 n_active -= 1
#     return idxs[active]

# def IoU(boxA, boxB):
#     ax, ay, aw, ah = boxA
#     bx, by, bw, bh = boxB
#     areaA = aw * ah
#     areaB = bw * bh
#     if areaA <= 0 or areaB <= 0: return 0
#     intersectionArea = (
#         max(min(ay + ah, by + bh) - max(ay, by), 0) * 
#         max(min(ax + aw, bx + bw) - max(ax, bx), 0))
#     return intersectionArea / (areaA + areaB - intersectionArea)


# def image_loader(im, imsize=640):
#     img = cv2.resize(im, (imsize,imsize))#int(imsize*im.shape[1]/im.shape[0])
#     img = img.transpose(2, 0, 1) #[:, :, ::-1]
#     img = torch.from_numpy(np.ascontiguousarray(img)).float()
#     img /= 255.0
#     return img[None]


class Tracker2(Tracker):
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        from detectron2.data import MetadataCatalog
        self.metadata = MetadataCatalog.get(f'tracker{id(self)}')
        self.metadata.thing_classes = []
        self.metadata.thing_colors = []
        super().__init__(metric, max_iou_distance, max_age, n_init)
    def _initiate_track(self, detection):
        track = super()._initiate_track(detection)
        self.metadata.thing_classes.append(f'track {self.tracks[-1].track_id}')
        self.metadata.thing_colors.append(random_color(rgb=True, maximum=1))
        return track

def run(src, max_cosine_distance=0.2, nn_budget=None, 
        out_file=None, fps=10, show=None, **kw):
    """Run multi-target tracker on a particular sequence.
    
    Arguments:
        min_confidence (float): Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
        nms_max_overlap (float): Maximum detection overlap (non-maxima suppression threshold).
        min_height (int): Detection height threshold. Disregard all detections that have a height lower than this value.
        max_cosine_distance (float): Gating threshold for cosine distance metric (object appearance).
        nn_budget (int): Maximum size of the appearance descriptor gallery. If None, no budget is enforced.
    """
    tracker = Tracker2(nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))

    # # model = Yolo(**kw)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    # model.eval()
    model = Detic(**kw)

    if out_file is True:
        out_file='tracked_'+os.path.basename(src)
    outsize=600

    with ImageOutput(out_file, fps, show=show) as imout:
        for i, im in video_feed(src, fps):
            im = cv2.resize(im, (outsize, int(outsize*im.shape[0]/im.shape[1])))
            # X_im = image_loader(im)
            # xywh, scores, features = model(im)
            outputs = model(im)
            # result.render()
            # imout.output(result.ims[0])
            print({k: getattr(v, 'shape', v) for k, v in outputs['instances'][0]._fields.items()})

            imout.output(model.draw(im, outputs))

            # imout.output(draw_boxes(
            #     im, 
            #     xywh, 
            #     [str(s) for s in scores]))

            # Update tracker.
            tracker.predict()
            tracker.update([
                Detection(r.pred_boxes.tensor, r.scores[0], r.clip_features[0])
                for r in outputs['instances']
            ])

            tracks = [
                track for track in tracker.tracks 
                if not track.is_confirmed() or track.time_since_update > 0
            ]

            v = Visualizer(im[:, :, ::-1], tracker.metadata, instance_mode=ColorMode.SEGMENTATION)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            im = out.get_image()[:, :, ::-1]

            # # imout.output(im)
            # imout.output(draw_boxes(
            #     im, 
            #     [d.to_tlwh() for d in tracks], 
            #     [str(d.track_id) for d in tracks]))


if __name__ == '__main__':
    import fire
    fire.Fire(run)
