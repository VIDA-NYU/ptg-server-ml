import collections
import os
import tqdm

import cv2
import numpy as np
import torch
from torch import nn

from ptgprocess.clip import ZeroClip
from ptgprocess.detic import Detic, BUILDIN_CLASSIFIER
from ptgprocess.util import video_feed, VideoInput, ImageOutput, draw_boxes
from detic_augmentors import get_augmentation

class VideoFrameInput:
    def __init__(self, root, start_frame=None, stop_frame=None):
        self.root = root
        self.start_frame = start_frame
        self.stop_frame = stop_frame

    def __enter__(self): return self
    def __exit__(self, *a): pass

    def __iter__(self):
        fs = sorted(os.listdir(self.root))
        start, stop = self.start_frame, self.stop_frame
        for f in tqdm.tqdm(fs):
            i = int(os.path.splitext(f)[0])
            if start and i < start:
                continue
            if stop and i > stop:
                break

            yield i, cv2.imread(os.path.join(self.root, f))


class Extract:
    def __init__(self, vocab, ann_root='epic-kitchens-100-annotations-normalized', include=None, exclude=None, splitby=None, **kw):
        self.vocab_name = vocab.replace(':','') if isinstance(vocab, str) else ''.join(vocab or []).replace(' ','')[:10]
        self.detic = Detic(vocab=['cat'], **kw)   # random vocab for faster init
        self.clip = ZeroClip()
        print("labels:", len(self.detic.labels), self.detic.labels)
        self.augmentors = None

    def get_out_dir(self, out_dir, video_name):
        if out_dir is True:
            out_dir = 'output/predictions'
        if out_dir:
            participant_id = video_name.split('_')[0] if video_name.startswith('P') else 'non-EK'
            out_dir = os.path.join(out_dir, participant_id, video_name)
            print('writing predictions to', out_dir)
            os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _dump(self, outputs, Z_clip, im):
        h, w = im.shape[:2]
        insts = outputs["instances"].to("cpu")
        boxes = insts.pred_boxes.tensor.numpy()
        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h

        i_sort = sorted(np.arange(len(boxes)), key=lambda i: tuple(boxes[i,:2]))
        class_ids = insts.pred_classes.numpy()[i_sort]
        return dict(
            boxes=boxes[i_sort], 
            frame_embed=Z_clip.cpu().numpy(),
            labels=np.array([self.detic.labels[i] for i in class_ids]),
            class_ids=class_ids, 
            clip_embeds=insts.clip_features.numpy()[i_sort],
            scores=insts.scores.numpy()[i_sort])

    def get_fname(self, out_dir, i, narr_id=None, j=None):
        return os.path.join(out_dir, f'{i:07d}{f"_{narr_id}" if narr_id is not None else ""}{f"_{j}" if j is not None else ""}.npz')

    def _compute(self, im, aug, out_dir, i, narr_id, j=None):
        fname = self.get_fname(out_dir, i, narr_id, j)
        if not os.path.isfile(fname):
            # compute model
            imi = aug(im) if aug else im
            outputs = self.detic(imi)
            Z_clip = self.clip.encode_image(imi)
            # save
            print({k: v.shape for k, v in self._dump(outputs, Z_clip, imi).items()})
            # np.savez(fname, **self._dump(outputs, Z_clip, imi))

    def _run(self, src, vocab, out_dir, narr_id=None, augs=(), imsize=600, start_frame=None, stop_frame=None):
        self.detic.set_vocab(vocab)

        video_name = os.path.splitext(os.path.basename(src))[0]
        out_dir = self.get_out_dir(out_dir, video_name)
        assert os.path.isdir(src)

        with VideoInput(src, start_frame=start_frame, stop_frame=stop_frame) as vid:
            for i, im in vid:
                # compute normal frame
                self._compute(im, None, out_dir, i, narr_id)
                # compute augmentations
                for j, aug in enumerate(augs or ()):
                    self._compute(im, aug, out_dir, i, narr_id, j)

    def run(self, *video_paths, out_dir, augment_csv=None):
        for v in video_paths:
            self._run(v, None, out_dir)

    def run_augs(self, 
            video_root='/vast/irr2020/EPIC-KITCHENS/videos', 
            augment_csv='/vast/bs3639/epic-kitchens-augment-src.csv', 
            ann_root='/vast/bs3639/epic-kitchens-100-annotations-normalized', 
            out_dir='/vast/bs3639/epic-kitchens-detic-clip', 
            verb_count=97, noun_count=293, desired_total=1e6,
            include=None, exclude=None, splitby=None,
            narration_key='gen_noun', **kw
        ):
        desired_verb_count = desired_total / verb_count
        desired_noun_count = desired_total / noun_count
        print('desired_verb_count', desired_verb_count)
        print('desired_noun_count', desired_noun_count)

        # load the annotation df
        import pandas as pd
        augment_df = pd.read_csv(augment_csv)
        augment_df = augment_df.drop(['frame_id', 'Unnamed: 0'], axis=1, errors='ignore').drop_duplicates()
        
        # describe the src df
        print(augment_df.shape)
        print(augment_df.head())
        print(augment_df.columns)
        print('verb_count', augment_df.verb_count.min(), augment_df.verb_count.max())
        print('min_noun_count', augment_df.min_noun_count.min(), augment_df.min_noun_count.max())

        # get the augmentation count
        augment_verb_count = np.ceil(desired_verb_count / augment_df.verb_count).astype(int)
        augment_noun_count = np.ceil(desired_noun_count / augment_df.min_noun_count).astype(int)
        augment_df['max_augment_count'] = pd.concat([augment_verb_count, augment_noun_count], axis=1).max(axis=1) - 1

        # show the statistics
        counts = augment_df.max_augment_count.value_counts()
        print(counts)
        print((counts.index.values * counts.values).sum())
        print(pd.Series(counts.index.values * counts.values))

        # filter out things we dont need to annotate
        augment_df = augment_df[augment_df.max_augment_count > 0]

        # get the full narration set
        action_df = pd.concat([
            pd.read_csv(os.path.join(ann_root, "EPIC_100_train_normalized.csv")).assign(split='train'),
            pd.read_csv(os.path.join(ann_root, "EPIC_100_validation_normalized.csv")).assign(split='val'),
        ])

        for i, row in augment_df.iterrows():
            # get the name of the video directory
            src = os.path.join(video_root, row.video_id.split('_')[0], row.video_id)

            # get the vocab list for the video
            vid_action_df = action_df[action_df.video_id == row.video_id]
            vocab = vid_action_df[narration_key].unique().tolist()
            if splitby:
                vocab = [x.strip() for x in vocab for x in x.split(splitby)]
            if exclude:
                vocab = [x for x in vocab if x not in exclude]
            if include:
                vocab = list(vocab)+list(include)
            vocab = list(set(vocab))

            augs = [
                get_augmentation()
                for i in range(int(row.max_augment_count))
            ]
            
            # compute the augmentation
            self._run(
                src, vocab, out_dir, augs=augs, 
                narr_id=row.narration_id, 
                start_frame=row.start_frame, 
                stop_frame=row.stop_frame, **kw)



VERB_COUNT = 91
NOUN_COUNT = 293

if __name__ == '__main__':
    import fire
    fire.Fire(Extract)