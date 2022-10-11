import os
from collections import deque
import numpy as np
import h5py
import cv2

def run(src, data_dir, out_dir, n_frames=16, fps=30, **kw):
    from ptgprocess.egovlp import EgoVLP
    from ptgprocess.util import VideoInput, VideoOutput, draw_text_list, get_vocab
    model = EgoVLP(**kw)

    out_file = get_out_file(src, data_dir, out_dir)
    print(out_file)

    q = deque(maxlen=n_frames)

    # compute
    i_frames = []
    results = []
    with VideoInput(src, fps, give_time=False) as vin:
        for j, (i, im) in enumerate(vin):
            im = cv2.resize(im, (600, 400))
            q.append(model.prepare_image(im))
            z_video = model.encode_video(torch.stack(list(q))).detach().cpu().numpy()
            i_frames.append(i)
            results.append(z_video)

    # save
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with h5py.File(out_file, 'a') as hf:
        i = np.array(i_frames)
        Z = np.concatenate(results)
        data = np.zeros(len(i_frames), dtype=[('frame', 'i', i.shape[1:]), ('Z', 'f', Z.shape[1:])])
        data['frame'] = i
        data['Z'] = Z

        dset_name = f'egovlp-n={n_frames}-fps={fps}'
        hf.create_dataset(dset_name, data=data)
    print('saved', out_file, dset_name, len(data))

def get_out_file(f, data_dir, out_dir):
    return os.path.join(out_dir, os.path.relpath(os.path.splitext(f)[0]+'.h5', data_dir))


if __name__ == '__main__':
    import fire
    fire.Fire(run)
