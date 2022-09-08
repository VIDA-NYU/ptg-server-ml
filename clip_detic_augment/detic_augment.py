import collections
import os
import tqdm

import cv2
import numpy as np
import torch
from torch import nn

# from ptgprocess.detic import Detic, BUILDIN_CLASSIFIER
from ptgprocess.util import video_feed, VideoInput, ImageOutput, draw_boxes


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



def get_vocab(vocab, ann_root, include=(), exclude=(), splitby=','):
    def _get(vocab):
        if vocab is None:
            return vocab  # someone elses problem lol
        if isinstance(vocab, (list, tuple)):
            return vocab  # literal
        if vocab in BUILDIN_CLASSIFIER:
            return vocab  # builtin
        if ':' in vocab:
            kind, vocab, key = vocab.split(':', 2)
            kind = kind.lower()
            if kind == 'recipe':
                import ptgctl
                api = ptgctl.API()
                recipe = api.recipes.get(vocab)
                return [w for k in key.split(',') for w in recipe[k]]
            if kind.startswith('ek'):
                import pandas as pd
                df = pd.concat([
                    pd.read_csv(os.path.join(ann_root, "EPIC_100_train_normalized.csv")).assign(split='train'),
                    pd.read_csv(os.path.join(ann_root, "EPIC_100_validation_normalized.csv")).assign(split='val'),
                ])
                df = df[df.video_id == vocab] if vocab != 'all' else df
                return df[key].unique().tolist()
        raise ValueError("Invalid vocab")

    vocab = _get(vocab)
    if isinstance(vocab, (list, tuple)):
        if splitby:
            vocab = [x.strip() for x in vocab for x in x.split(splitby)]
        if exclude:
            vocab = [x for x in vocab if x not in exclude]
        if include:
            vocab = list(vocab)+list(include)
        vocab = list(set(vocab))
    return vocab



class Extract:
    def __init__(self, vocab, ann_root='epic-kitchens-100-annotations-normalized', include=None, exclude=None, splitby=None, **kw):
        self.vocab_name = vocab.replace(':','') if isinstance(vocab, str) else ''.join(vocab or []).replace(' ','')[:10]
        vocab = get_vocab(vocab, ann_root, include, exclude, splitby)
        self.model = Detic(vocab=vocab, **kw)
        print("labels:", len(self.model.labels), self.model.labels)
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

    def _dump(self, outputs, im):
        h, w = im.shape[:2]
        insts = outputs["instances"].to("cpu")
        boxes = insts.pred_boxes.tensor.numpy()
        boxes[:, 0] /= w
        boxes[:, 1] /= h
        boxes[:, 2] /= w
        boxes[:, 3] /= h
        return dict(
            boxes=boxes, 
            class_ids=insts.pred_classes.numpy(), 
            clip_embeds=insts.clip_features.numpy(),
            scores=insts.scores.numpy())

    def get_fname(self, out_dir, i, j=None):
        return os.path.join(out_dir, f'{i:07d}{f"_{j}" if j is not None else ""}.npz')

    def _compute(self, im, aug, out_dir, i, j=None):
        # compute normal frame
        fname = self.get_fname(out_dir, i)
        if not os.path.isfile(fname):
            imi = aug(im) if aug else im
            np.savez(fname, self._dump(self.model(imi), imi))

    def _run(self, src, out_dir, augs=(), imsize=600, start_frame=None, stop_frame=None):
        video_name = os.path.splitext(os.path.basename(src))[0]

        out_dir = self.get_out_dir(out_dir, video_name)
        assert os.path.isdir(src)
        with VideoFrameInput(src, start_frame, stop_frame) as vid:
            for i, im in vid:
                h, w = int(imsize*im.shape[0]/im.shape[1]), int(imsize)

                # compute normal frame
                self._compute(im, None, out_dir, i)
                for j, aug in enumerate(augs):
                    self._compute(im, aug, out_dir, i, j)

    def run(self, *video_paths, out_dir, augment_csv=None):
        for v in video_paths:
            self._run(v, out_dir)

    def run_augs(self, video_root, augment_csv, out_dir, verb_count=97, noun_count=293, desired_total=1e6):
        desired_verb_count = desired_total / verb_count
        desired_noun_count = desired_total / noun_count
        print('desired_verb_count', desired_verb_count)
        print('desired_noun_count', desired_noun_count)

        import pandas as pd
        augment_df = pd.read_csv(augment_csv)
        augment_df = augment_df.drop(['frame_id', 'Unnamed: 0'], axis=1, errors='ignore').drop_duplicates()
        print(augment_df.shape)
        print(augment_df.head())
        print(augment_df.columns)
        print('verb_count', augment_df.verb_count.min(), augment_df.verb_count.max())
        print('min_noun_count', augment_df.min_noun_count.min(), augment_df.min_noun_count.max())

        augment_verb_count = np.ceil(desired_verb_count / augment_df.verb_count).astype(int)
        augment_noun_count = np.ceil(desired_noun_count / augment_df.min_noun_count).astype(int)
        augment_df['max_augment_count'] = pd.concat([augment_verb_count, augment_noun_count], axis=1).max(axis=1) - 1

        counts = augment_df.max_augment_count.value_counts()
        print(counts)
        print((counts.index.values * counts.values).sum())
        print(pd.Series(counts.index.values * counts.values))

        # augment_df = augment_df[augment_df.max_augment_count > 0]
        # for i, row in augment_df.iterrows():
        #     src = os.path.join(video_root, row.video_id.split('_')[0], row.video_id)
        #     self._run(src, out_dir, start_frame=row.start_frame, stop_frame=row.stop_frame)



VERB_COUNT = 91
NOUN_COUNT = 293

if __name__ == '__main__':
    import fire
    fire.Fire(Extract)