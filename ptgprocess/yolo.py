import os
import ultralytics
import numpy as np
import torch
from torch import nn

import gdown
import logging
from ultralytics.yolo.utils import LOGGER as _UL_LOGGER
_UL_LOGGER.setLevel(logging.WARNING)

MODEL_DIR = os.getenv('MODEL_DIR') or 'models'
DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, 'yolo_bbn.pt')

MODELS = {
    "M2": "1hnX9XhVPGhXTLMQfLK7rDponHf77KH61",
    "M5": "1PQyhoCbDNkqdm3BpwiWywAngnIrCpp26",
    "R18": "1F66I5f4J1_jTzkJc_ZTaJBXEBkxly90f",
}

SKILLS_CHECKPOINTS = { 
    k: os.path.join(MODEL_DIR, f'bbn_yolo_{k}.pt') 
    for k in MODELS 
}
for k, gid in MODELS.items():
    if not os.path.isfile(SKILLS_CHECKPOINTS[k]):
        gdown.download(id=gid, output=SKILLS_CHECKPOINTS[k])

SKILLS_CHECKPOINTS['tourniquet'] = SKILLS_CHECKPOINTS['M2']
SKILLS_CHECKPOINTS['chestseal'] = SKILLS_CHECKPOINTS['R18']
SKILLS_CHECKPOINTS['xstat'] = SKILLS_CHECKPOINTS['M5']

class BBNYolo(ultralytics.YOLO):
    def __init__(self, skill='tourniquet',  **kw) -> None:
        checkpoint = SKILLS_CHECKPOINTS[skill]
        super().__init__(checkpoint, **kw)
        names: dict = self.names
        self.labels = np.array([str(names.get(i, i)) for i in range(len(names))])

    # def forward(self, im):
    #     results = super().forward(im)
    #     boxes = results[0].boxes
        
    #     return 
    
    def unpack_results(self, results, box_type='xyxyn'):
        boxes = results[0].to("cpu").boxes
        cls_ids = boxes.cls.numpy().astype(int)
        return getattr(boxes, box_type).numpy(), None, cls_ids, self.labels[cls_ids], boxes.conf.numpy(), None


def as_objects(model, xyxy, class_ids, confs, ts):
    objects = []
    for xy, c, cid in zip(xyxy, confs, class_ids):
        objects.append({
            "xyxyn": xy.tolist(),
            "confidence": c,
            "class_id": cid,
            "label": model.labels[cid],
            "ts": ts,
        })
    return objects

def as_v2_objs(model, xyxy, confs, class_ids, labels, ious):
    objects = []
    for xy, c, cid, l, iou in zip(xyxy, confs, class_ids, labels, ious):
        # make object
        objects.append({
            "xyxyn": xy.tolist(),
            "class_ids": [cid],
            "labels": [l],
            "confidences": [c],
            # "box_confidences": [boxc],
            "hoi_iou": iou,
        })
    return objects

@torch.no_grad()
def run(src, out_file=None, fps=16, n_frames=32, stride=8, show=None):
    import cv2
    from ptgprocess.util import VideoInput, VideoOutput, draw_boxes
    model = BBNYolo()
    

    if out_file is True:
        out_file='bbn_yolo_'+os.path.basename(src)

    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        topk = []
        for i, (t, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            out = model(im)
            
            imout.output(draw_boxes(im, xyxy, model.labels)[0])
