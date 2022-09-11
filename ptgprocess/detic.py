from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import sys
import numpy as np
import torch
from torch import nn

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, random_color, ColorMode
from detectron2.data import MetadataCatalog

# Detic libraries
detic_path = os.getenv('DETIC_PATH') or 'Detic'
sys.path.insert(0,  detic_path)
sys.path.insert(0, os.path.join(detic_path, 'third_party/CenterNet2'))
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder
from centernet.config import add_centernet_config


BUILDIN_CLASSIFIER = {
    'lvis':       os.path.join(detic_path, 'datasets/metadata/lvis_v1_clip_a+cname.npy'),
    'objects365': os.path.join(detic_path, 'datasets/metadata/o365_clip_a+cnamefix.npy'),
    'openimages': os.path.join(detic_path, 'datasets/metadata/oid_clip_a+cname.npy'),
    'coco':       os.path.join(detic_path, 'datasets/metadata/coco_clip_a+cname.npy'),
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}


DEFAULT_PROMPT = 'a {}'

class Detic(nn.Module):
    def __init__(self, vocab=None, conf_threshold=0.3, masks=False, patch_for_embeddings=True, prompt=DEFAULT_PROMPT):
        super().__init__()
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(os.path.join(detic_path, "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"))
        cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
        cfg.MODEL.MASK_ON = masks
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = os.path.join(detic_path, cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        # print(cfg)
        self.predictor = DefaultPredictor(cfg)

        if patch_for_embeddings:
            self.predictor.model.roi_heads.__class__ = DeticCascadeROIHeads2
            for b in self.predictor.model.roi_heads.box_predictor:
                b.__class__ = DeticFastRCNNOutputLayers2
                b.cls_score.__class__ = ZeroShotClassifier2
        
        for cascade_stages in range(len(self.predictor.model.roi_heads.box_predictor)):
            self.predictor.model.roi_heads.box_predictor[cascade_stages].test_score_thresh = conf_threshold

        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()

        self.set_vocab(vocab, prompt)
        
    def set_vocab(self, vocab, prompt=DEFAULT_PROMPT):
        if isinstance(vocab, (list, tuple)):
            self.vocab_key = '__vocab:' + ','.join(vocab)
            self.metadata = metadata = MetadataCatalog.get(self.vocab_key)
            try:
                metadata.thing_classes = list(vocab)
                metadata.thing_colors = [tuple(random_color(rgb=True, maximum=1)) for _ in metadata.thing_classes]
            except (AttributeError, AssertionError):
                pass

            self.prompt = prompt = prompt or '{}'
            self.text_features = classifier = self.text_encoder(
                [prompt.format(x) for x in vocab]
            ).detach().permute(1, 0).contiguous().cpu()
        else:
            vocab = vocab or 'lvis'
            self.vocab_key = BUILDIN_METADATA_PATH[vocab]
            self.metadata = metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocab])
            classifier = BUILDIN_CLASSIFIER[vocab]    
        
        self.labels = np.asarray(metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, len(metadata.thing_classes))

    def forward(self, im):
        return self.predictor(im)

    def draw(self, im, outputs):
        v = Visualizer(im[:, :, ::-1], self.metadata, instance_mode=ColorMode.SEGMENTATION)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]


# disable jitter
def _jitter(self, c):
    return [c*255 for c in c]
Visualizer._jitter = _jitter


from torch.nn import functional as F
from detic.modeling.roi_heads.detic_roi_heads import DeticCascadeROIHeads
from detic.modeling.roi_heads.detic_fast_rcnn import DeticFastRCNNOutputLayers
from detic.modeling.roi_heads.zero_shot_classifier import ZeroShotClassifier
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

class DeticCascadeROIHeads2(DeticCascadeROIHeads):
    def _forward_box(self, features, proposals, targets=None, ann_type='box', classifier_info=(None,None,None)):
        if self.mult_proposal_score:
            k='scores' if len(proposals) > 0 and proposals[0].has('scores') else 'objectness_logits'
            proposal_scores = [p.get(k) for p in proposals]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(
                    prev_pred_boxes, image_sizes,
                    logits=[p.objectness_logits for p in proposals])
            predictions = self._run_stage(features, proposals, k, 
                classifier_info=classifier_info)  # added x_features as i=2
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))
        # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
        scores_per_stage = [h[0].predict_probs(h[1][:2]+h[1][3:], h[2]) for h in head_outputs] # ++ remove x_features from h
        scores = [
            sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
            for scores_per_image in zip(*scores_per_stage)
        ]
        if self.mult_proposal_score:
            scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
        if self.one_class_per_proposal:
            scores = [s * (s == s[:, :-1].max(dim=1)[0][:, None]).float() for s in scores]
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes((predictions[0], predictions[1]), proposals)
        pred_instances, filt_idxs = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        # ++ add clip features to instances [N boxes x 512]
        pred_instances[0].clip_features = predictions[2][filt_idxs]
        return pred_instances


class DeticFastRCNNOutputLayers2(DeticFastRCNNOutputLayers):
    def forward(self, x, classifier_info=(None,None,None)):
        """
        enable classifier_info
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = []
        cls_scores, x_features = self.cls_score(x, classifier=classifier_info[0])  # ++ add x_features
        scores.append(cls_scores)
   
        if classifier_info[2] is not None:
            cap_cls = classifier_info[2]
            cap_cls = cap_cls[:, :-1] if self.sync_caption_batch else cap_cls
            caption_scores, _ = self.cls_score(x, classifier=cap_cls)
            scores.append(caption_scores)
        scores = torch.cat(scores, dim=1) # B x C' or B x N or B x (C'+N)

        proposal_deltas = self.bbox_pred(x)
        if self.with_softmax_prop:
            prop_score = self.prop_score(x)
            return scores, proposal_deltas, x_features, prop_score
        else:
            return scores, proposal_deltas, x_features
        # ++ return x_features


class ZeroShotClassifier2(ZeroShotClassifier):
    def forward(self, x, classifier=None):
        '''
        Inputs:
            x: B x D'
            classifier_info: (C', C' x D)
        '''
        x_features = x = self.linear(x)  # ++ save linear output
        if classifier is not None:
            zs_weight = classifier.permute(1, 0).contiguous() # D x C'
            zs_weight = F.normalize(zs_weight, p=2, dim=0) \
                if self.norm_weight else zs_weight
        else:
            zs_weight = self.zs_weight
        if self.norm_weight:
            x = self.norm_temperature * F.normalize(x, p=2, dim=1)
        x = torch.mm(x, zs_weight)
        if self.use_bias:
            x = x + self.cls_bias
        return x, x_features  # ++ add x_features



def chunks(iterable,size):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it,size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it,size))


import itertools

class VideoFramesLoader:
    def __init__(self, root_dir, batch_size=64, ext='*'):
        self.root = root_dir
        self.ext = ext

    def __iter__(self):
        fs = glob.iglob(os.path.join(self.root, f'*.{self.ext}'))
        for chunk in chunks(fs):
            ims = []




def video_batch_compute(vid_dir):
    # 

    # load vocab

    model = Detic()

    fs = glob.glob(os.path.join(self.root, f'*.{self.ext}'))

    for fs_batch, im_batch in VideoFramesLoader(vid_dir):
        fs_batch = model(im_batch)






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
        for i, im in video_feed(src, fps):
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
