import os
import collections
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import clip
from PIL import Image
import matplotlib.pyplot as plt


def time_distributed(model, x):
    xr = x.contiguous().view(-1, *x.shape[2:])
    y = model(xr)
    return y.contiguous().view(*x.shape[:2], *y.shape[1:])


class ActionCLIP(nn.Module):
    def __init__(self, checkpoint=None, time_pool=True, device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self._transform = clip.load("ViT-B/32", device=self.device)
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location=self.device)['state_dict']
            self.load_state_dict({
                k: v for k, v in state_dict.items() 
                if not k.startswith('teacher.') and k not in {'sink_temp'}
            })
        self.time_pool = time_pool
        self.dtype = self.model.dtype

    def forward(self, images, text):
        Z_images = self.encode_image(images)
        Z_text = self.encode_text(text)
        similarity = Z_images @ Z_text.t()
        return similarity

    def preprocess_images(self, images):  # [time, h,w,c] => [1, time, ...]
        images = torch.stack([
            self._transform(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in images
        ]).to(self.device)[None]
        return images

    def preprocess_text(self, text):
        return torch.cat([clip.tokenize(t, truncate=True) for t in text]).to(self.device)

    def encode_images(self, images):
        return F.normalize(
            time_distributed(self.model.encode_image, images) 
            if self.time_pool else
            self.model.encode_image(images), 
            dim=1)

    def encode_text(self, text):
        model = self.model
        x = model.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return F.normalize(x, dim=1)

    # def get_accuracy_topk(self, images, text, text_truth, k=1):
    #     similarity = self(images, text)
    #     predictions_topk = torch.topk(similarity, k, dim=-1)[1]
    #     accuracy_topk = torch.mean(text_truth == predictions_topk)
    #     return accuracy_topk


def load_action_file(action_file):
    actions = open(action_file, 'r').read().splitlines()
    deduped = []
    for x in actions:
        if x not in deduped:
            deduped.append(x)
    return deduped

import tqdm
def _video_feed(src=0, fps=None):
    import cv2
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    every = int(src_fps/fps) if fps else 1
    i = 0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm.tqdm(total=total)
    while True:
        ret, im = cap.read()
        i += 1
        pbar.update()
        if not ret:
            break
        if i%every: 
            continue
        yield i / fps, im
        

def _pool_frames(it, n, hop=None):
    hop = hop or n
    q = collections.deque(maxlen=n)
    i = 0
    for im in it:
        q.append(im)
        i += 1
        if i >= hop and len(q) >= n:
            yield q
            i = 0


def draw_text_list(img, texts, i=0, tl=(10, 50), scale=0.5, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX, 
            scale, color, thickness)
    return img



class ImageOutput:
    def __init__(self, src, fps, cc='avc1', show=None):
        self.src = src
        self.cc = cc
        self.fps = fps
        self._show = not src if show is None else show

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        if self._show:
            cv2.destroyAllWindows()

    def output(self, im):
        if self.src:
            self.write_video(im)
        if self._show:
            self.show_video(im)

    _w = None
    def write_video(self, im):
        if not self._w:
            self._w = cv2.VideoWriter(
                self.src, cv2.VideoWriter_fourcc(*self.cc),
                self.fps, im.shape[:2][::-1], True)
        self._w.write(im)

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


class CsvWriter:
    def __init__(self, fname, header):
        self.fname = fname
        self.header = header

    def __enter__(self): return self
    def __exit__(self, *a):
        if self._f:
            self._f.close()
        self._w = self._f = None
    
    _w = _f = None
    def write(self, row):
        if not self._w:
            import csv
            self._f = open(self.fname, 'w')
            self._w = csv.writer(self._f)
            if self.header:
                self._w.writerow(self.header)
        self._w.writerow([self.format(x) for x in row])

    def format(self, x):
        if isinstance(x, float):
            return float(f'{x:.4f}')
        return x



localfile = lambda *fs: os.path.join(os.path.dirname(__file__), *fs)
CHECKPOINT = localfile('models/epoch=2-step=99021.ckpt')
ACTION_FILE = localfile('actions/all_actions.txt')

import datetime
def main(src=0, action_file=ACTION_FILE, out_file=None, csv_file=None, show=None, checkpoint=CHECKPOINT, nframes=10, fps=10, topk=5, output_dir='output'):
    output_dir = f'{output_dir}/outputs-{datetime.datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    src_str = f'webcam-{src}.mp4' if isinstance(src, int) else src
    if out_file is True:
        out_file = os.path.join(output_dir, f'actionclip-{os.path.basename(src_str)}')
    if csv_file is True:
        csv_file = os.path.join(output_dir, f'actionclip-{os.path.basename(src_str)}.csv')
    print("loading model")
    model = ActionCLIP(checkpoint)
    print('loaded model')

    actions = load_action_file(action_file)
    print("Loaded actions", len(actions))
    Z_text = model.encode_text(model.preprocess_text(actions)).t()
    print("Loaded text embeddings", Z_text.shape)

    with open(os.path.join(output_dir, 'actions.txt'), 'w') as f:
        f.write('\n'.join(actions))

    from pyinstrument import Profiler
    profile = Profiler()
    try:
        cmap = plt.get_cmap('magma')
        zimq = collections.deque(maxlen=nframes)
        simq = collections.deque(maxlen=400)
        with profile, ImageOutput(out_file, fps, show=show) as imout, CsvWriter(csv_file, ["time"]+actions) as csvout:
            for t, im in _video_feed(src, fps=fps):
                Z_images = model.encode_images(model.preprocess_images([im]))[0]
                zimq.append(Z_images)

                Z_images = torch.stack(tuple(zimq)).mean(dim=0)
                similarity = Z_images @ Z_text
                similarity_soft = similarity.softmax(dim=-1)
                predictions_topk = torch.topk(similarity_soft, topk, dim=-1)[1][0]

                similarity_soft = similarity_soft.detach().numpy()
                simq.append(similarity_soft)
                sim = (cmap(np.concatenate(list(simq)))[:,:,:3] * 255).astype('uint8')
                sim = cv2.resize(sim, (im.shape[1], 150))
                
                pred_actions = [f'{actions[i]} ({similarity_soft[0, i]:.0%})' for i in predictions_topk]
                tqdm.tqdm.write('  |  '.join(pred_actions[:4]))
                im = draw_text_list(im, pred_actions)
                imout.output(np.concatenate([im, sim], axis=0))
                csvout.write([t, *similarity_soft.squeeze().tolist()])
    except KeyboardInterrupt:
        print("\nk bye! :)")
    finally:
        profile.print()

# ANN_FILE = 'epic-kitchens-100-annotations-normalized/EPIC_100_validation_normalized.csv'
# import csv
# def get_unique_video_actions(annotation_file=ANN_FILE, participant=None, video=None):
#     actions = []
#     with open(annotation_file, newline='') as csvfile:
#         reader = csv.reader(csvfile,delimiter=',')
#         next(reader,None)
#         for row in reader:
#             if participant and row[1] != participant or video and row[2] != video:
#                 continue
#             actions.append(row[8])
#     return sorted(set(actions))



if __name__ == '__main__':
    import fire
    fire.Fire(main)