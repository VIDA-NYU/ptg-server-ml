import os
from collections import deque
import numpy as np
import h5py
import cv2
import torch

def run(src, data_dir='.', out_dir='.', n_frames=16, fps=30, overwrite=False, **kw):
    out_file = get_out_file(src, data_dir, out_dir)
    dset_name = f'egovlp-n={n_frames}-fps={fps}'
    print(out_file, dset_name)

    if not overwrite and os.path.isfile(out_file):
        with h5py.File(out_file, 'a') as hf:
            if dset_name in hf:
                return

    from ptgprocess.egovlp import EgoVLP
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list, get_vocab
    model = EgoVLP(**kw)

    q = deque(maxlen=n_frames)

    # compute
    i_frames = []
    results = []
    with VideoInput(src, fps, give_time=False) as vin:
        for j, (i, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.append(model.prepare_image(im))
            z_video = model.encode_video(torch.stack(list(q), dim=1).cuda()).detach().cpu().numpy()
            assert len(z_video) == 1, 'batch size should be one'
            i_frames.append(i)
            results.append(z_video[0])

    # save
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with h5py.File(out_file, 'a') as hf:
        i = np.array(i_frames)
        Z = np.array(results)
        print(i.shape, {z.shape for z in Z})
        data = np.zeros(len(i_frames), dtype=[('frame', 'i', i.shape[1:]), ('Z', 'f', Z.shape[1:])])
        data['frame'] = i
        data['Z'] = Z

        if dset_name in hf:
            del hf[dset_name]
        hf.create_dataset(dset_name, data=data)
    print('saved', out_file, dset_name, len(data))

def get_out_file(f, data_dir, out_dir):
    return os.path.join(out_dir, os.path.relpath(os.path.splitext(f)[0]+'.h5', data_dir))





# visualization


def vis(h5file, ann_file, video_id, name=None):
    with h5py.File(h5file, 'r') as hf:
        if name is None:
            print('pick one of:')
            print('\n'.join(hf))
            return
        d = hf[name][:]
        print(d.dtype)
        Z = d['Z']
        frames = d['frame']
    gt_plot(Z, frames, ann_file, video_id)

def get_action_df(df):
    import pandas as pd
    df = df[['narration','start_frame','stop_frame']].sort_values('start_frame')
    overlaps = df.start_frame <= df.stop_frame.shift().fillna(-1)
    noac_df = pd.DataFrame({
            'narration': 'no action',
            'start_frame': df.stop_frame.shift(fill_value=0)[~overlaps].values,
            'stop_frame': df.start_frame[~overlaps].values,
    }, index=df.index[~overlaps] - 0.5)
    return pd.concat([df, noac_df]).sort_index().reset_index(drop=True)

def gt_plot(Z, frames, ann_file, video_id):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(ann_file)
    df = df[df['video_id']==video_id]
    df = df[['narration','start_frame','stop_frame']].sort_values('start_frame')
    gt = ['no action']+[(df[(i >= df.start_frame) & (i < df.stop_frame)].narration.tolist() or ['no action'])[-1] for i in frames]
    actions, action_order, action_ix = np.unique(gt, return_index=True, return_inverse=True)
    actions = actions[np.argsort(action_order)]
    action_ix = np.argsort(np.argsort(action_order))[action_ix]


    action_ix = action_ix[1:]
    print(actions)

    import torch
    from ptgprocess.egovlp import EgoVLP, similarity

    model = EgoVLP()
    Z_text = model.encode_text(list(actions))
    y = similarity(Z_text, torch.Tensor(Z).cuda()).cpu().numpy()
    print(y.shape)
    print(df.shape, action_ix.shape)

    plt.figure(figsize=(12, 6))
    plt.imshow(y.T, aspect='auto', origin='lower')
    
    plt.plot(action_ix, c='r')
    plt.yticks(range(len(actions)), actions)
    plt.savefig(f'{video_id}_emissions.png')



if __name__ == '__main__':
    import fire
    fire.Fire()
