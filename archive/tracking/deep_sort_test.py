import os
from ptgprocess.detic import Detic
import deep_sort as ds



def run_video(src, vocab, ann_root=None, include=None, exclude=None, out_file=None, fps=10, show=None, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    from ptgprocess.util import VideoInput, VideoOutput, draw_boxes, get_vocab

    tracker = ds.Tracker(ds.NearestNeighbor('cosine', 0.2))

    model = Detic(**kw)

    if out_file is True:
        out_file='detic_'+os.path.basename(src)

    vocab = get_vocab(vocab, ann_root, include, exclude)
    assert vocab, 'you must set vocab'
    model.set_vocab(vocab)

    with VideoInput(src, fps, give_time=True) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        for t, im in vin:
            instances = model(im)['instances']
            tlwh = ds.util.tlbr2tlwh(instances.pred_boxes.tensor.numpy())
            scores = instances.scores
            features = instances.clip_features
            class_ids = instances.pred_classes
            labels = model.labels[class_ids]

            tracker.predict()
            tracker.update([
                ds.Detection(x, s, f, class_id=c, label=l, time=t)
                for x, s, f, c, l in zip(tlwh, scores, features, class_ids, labels)
            ])

            tlwh = [t.tlwh for t in tracker.tracks]
            labels = [f'{t.track_id}{"" if t.is_confirmed() else "?"}' for t in tracker.tracks]
            imout.output(draw_boxes(im, tlwh, labels))


if __name__ == '__main__':
    import fire
    fire.Fire(run_video)