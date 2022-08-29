from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import sys
import torch
from torch import nn

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Detic libraries
detic_path = '/opt/gh/Detic'
sys.path.insert(0,  detic_path)
sys.path.insert(0, os.path.join(detic_path, 'third_party/CenterNet2/'))
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
from centernet.config import add_centernet_config


BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}


DEFAULT_PROMPT = 'a {}'

class Detic(nn.Module):
    def __init__(self, vocab=None, conf_threshold=0.3, prompt=DEFAULT_PROMPT):
        super().__init__()
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(os.path.join(detic_path, "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"))
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
        cfg.MODEL.MASK_ON = False
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        # print(cfg)
        self.predictor = DefaultPredictor(cfg)
        for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
            self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = conf_threshold

        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

        if vocab:
            self.set_vocab(vocab)
        
    def set_vocab(self, vocab, prompt=DEFAULT_PROMPT):
        if isinstance(vocab, (list, tuple)):
            self.vocab_key = '__vocab:' + ','.join(vocab)
            self.metadata = metadata = MetadataCatalog.get(self.vocab_key)
            try:
                metadata.thing_classes = list(vocab)
            except AttributeError:
                pass

            self.prompt = prompt = prompt or '{}'
            classifier = self.text_encoder(
                [prompt.format(x) for x in vocab]
            ).detach().permute(1, 0).contiguous().cpu()
        else:
            self.vocab_key = BUILDIN_METADATA_PATH[vocab]
            self.metadata = metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocab])
            classifier = BUILDIN_CLASSIFIER[vocab]    
    
        reset_cls_test(self.predictor.model, classifier, len(metadata.thing_classes))

    def forward(self, im):
        return self.predictor(im)

    def draw(self, im, outputs):
        v = Visualizer(im[:, :, ::-1], self.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]



def run(src, vocab, max_cosine_distance=0.2, nn_budget=None, 
        out_file=None, fps=10, show=None):
    """Run multi-target tracker on a particular sequence.
    
    Arguments:
        min_confidence (float): Detection confidence threshold. Disregard all detections that have a confidence lower than this value.
        nms_max_overlap (float): Maximum detection overlap (non-maxima suppression threshold).
        min_height (int): Detection height threshold. Disregard all detections that have a height lower than this value.
        max_cosine_distance (float): Gating threshold for cosine distance metric (object appearance).
        nn_budget (int): Maximum size of the appearance descriptor gallery. If None, no budget is enforced.
    """
    # from deep_sort import nn_matching
    # from deep_sort.detection import Detection
    # from deep_sort.tracker import Tracker
    from ptgprocess.util import ImageOutput, video_feed, draw_boxes

    # tracker = Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget))

    model = Detic()

    if out_file is True:
        out_file='tracked_'+os.path.basename(src)

    assert vocab, 'you must set vocab'
    model.set_vocab(vocab)

    with ImageOutput(out_file, fps, show=show) as imout:
        for i, im in video_feed(src):
            # X_im,  im = image_loader(im)
            outputs = model(im)

            xywh = outputs["instances"].pred_boxes.tensor
            scores = outputs["instances"].scores
            # valid = scores >= min_confidence
            # xywh = xywh[valid]
            # scores = scores[valid]
            print(xywh.__dict__)
            print(xywh.shape)

            # xywh = xywh.clone()
            xywh[:,[2,3]] -= xywh[:,[0,1]]
            # xywh = xywh[(xywh[:,3] >= min_height)]
            # feature = None
            print(xywh)

            # # Update tracker.
            # tracker.predict()
            # tracker.update([
            #     Detection(xywh, scores, )
            #     for xywh in xywh
            # ])

            # tracks = [
            #     track for track in tracker.tracks 
            #     if not track.is_confirmed() or track.time_since_update > 0
            # ]

            labels = [
                model.metadata.thing_classes[x] 
                for x in outputs["instances"].pred_classes.cpu().tolist()
            ]
            imout.output(draw_boxes(im, xywh, labels))
            # imout.output(draw_boxes(
            #     im, 
            #     [d.to_tlwh() for d in tracks], 
            #     [d.track_id for d in tracks]))


if __name__ == '__main__':
    import fire
    fire.Fire(run)
