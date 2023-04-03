import os
import ultralytics
import numpy as np
import torch
from torch import nn

import logging
from ultralytics.yolo.utils import LOGGER as _UL_LOGGER
_UL_LOGGER.setLevel(logging.WARNING)

MODEL_DIR = os.getenv('MODEL_DIR') or 'models'
DEFAULT_CHECKPOINT = os.path.join(MODEL_DIR, 'yolo_bbn.pt')


class BBNYolo(ultralytics.YOLO):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT,  **kw) -> None:
        super().__init__(checkpoint, **kw)
        names: dict = self.names
        self.labels = np.array([str(names.get(i, i)) for i in range(len(names))])

    # def forward(self, im):
    #     results = super().forward(im)
    #     boxes = results[0].boxes
        
    #     return 
    
    def unpack_results(self, results):
        boxes = results[0].to("cpu").boxes
        cls_ids = boxes.cls.numpy().astype(int)
        return boxes.xyxyn.numpy(), None, cls_ids, self.labels[cls_ids], boxes.conf.numpy(), None