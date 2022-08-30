import os
import tqdm
import numpy as np
import pandas as pd

import cv2
import torch
from torch import nn

from ptgprocess.clip import MODELS
from ptgprocess.util import video_feed, draw_text_list
import ptgctl

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(24, 6))
import matplotlib.animation as pltanim
from adjustText import adjust_text


from sklearn.manifold import LocallyLinearEmbedding #Isomap, 
    # from sklearn.decomposition import PCA

ANN_ROOT='/opt/Github/epic-kitchens-100-annotations-normalized'

def draw(video_fname, model='action2', fps=10, tlim=None, text_col='narration', normalized=True, recipe=None, recipe_key='steps', checkpoint=None, ann_root=ANN_ROOT):
    model = MODELS[model](checkpoint)
    
    df = pd.concat([
        pd.read_csv(os.path.join(ann_root, f"EPIC_100_train{'_normalized' if normalized else ''}.csv")).assign(split='train'),
        pd.read_csv(os.path.join(ann_root, f"EPIC_100_validation{'_normalized' if normalized else ''}.csv")).assign(split='val'),
    ])
    ann_videos = df.video_id.unique()


    video_name = os.path.splitext(os.path.basename(video_fname))[0]
    out_dir = f'output/{video_name}-{model.__class__.__name__}'
    os.makedirs(out_dir, exist_ok=True)
    has_gt = video_name in ann_videos

    if recipe:
        import ptgctl
        api = ptgctl.API()
        texts = api.recipes.get(recipe)[recipe_key]
        has_gt=False
    else:
        df_vid = df[df.video_id == video_name]
        texts = df_vid[text_col].unique()



    Z_text = model.encode_text(texts).detach().numpy()

    ims = []
    for t, im in video_feed(video_fname, fps=fps):
        if tlim and t > tlim:
            break
        ims.append(cv2.resize(im, (760, 428)))
    ims = np.stack(ims)

    Z_images = []
    for im in tqdm.tqdm(ims):
        z_im = model.encode_image(im).detach()
        Z_images.append(z_im)
    Z_images = np.concatenate(Z_images)



    dimreduc = LocallyLinearEmbedding(n_components=2)
    Zt_text = dimreduc.fit_transform(Z_text)

    plt.figure(figsize=(16, 10))
    plt.scatter(Zt_text[:,0], Zt_text[:,1], label='text', c='r')
    ax=plt.gca()
    for z, t in zip(Zt_text, texts):
        ax.annotate(t, (z[0]+0.003, z[1]-0.002))
    plt.savefig(os.path.join(out_dir, 'text-space.jpg'))
    plt.close()

    Zt_images = dimreduc.transform(Z_images)

    plt.figure(figsize=(16, 10))
    plt.scatter(Zt_text[:,0], Zt_text[:,1], label='text', c='r')

    ax=plt.gca()
    text_annots = [ax.annotate(t, (z[0], z[1])) for z, t in zip(Zt_text, texts)]
    adjust_text(text_annots, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), lim=50)
    plt.plot(Zt_images[:,0], Zt_images[:,1])
    plt.scatter(Zt_images[:,0], Zt_images[:,1])
    plt.savefig(os.path.join(out_dir, 'text+image-space.jpg'))
    plt.close()





    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    ax1.axis('off')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()

    imshowed = ax1.imshow(ims[0])

    ax2.scatter(Zt_text[:,0], Zt_text[:,1], label='text', c='r', s=50)
    text_annots = [ax2.annotate(t, (z[0], z[1])) for z, t in zip(Zt_text, texts)]
    adjust_text(
        text_annots, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), 
        lim=50, avoid_self=False)

    ax2.plot(Zt_images[:,0], Zt_images[:,1], c='k', alpha=0.1)
    imscatter = ax2.scatter([], [], s=100)

    topk = 5
    lines_topk = [ax2.plot([], [], c='r')[0] for i in range(topk)]

    def animate(frame):
        im = ims[frame].copy()

        similarity = torch.Tensor(100*Z_images[frame] @ Z_text.T).softmax(dim=-1)
        sim = similarity.detach().tolist()

        i_topkmax = torch.topk(similarity, topk, dim=-1)[1].detach().cpu().numpy().astype(int)
        pred_labels = [f'{texts[i]} ({sim[i]:.0%})' for i in i_topkmax]
        pred_text = [texts[i] for i in i_topkmax]

        i=0
        if has_gt:
            dft = df_vid[(df_vid.start_frame/60 < frame/fps) & (df_vid.stop_frame/60 > frame/fps)]
            texts_true = list(dft.narration.unique())
            _, i = draw_text_list(im, [t for t in texts_true if t in pred_text[:1]], color=(0,255,0))
            _, i = draw_text_list(im, [t for t in texts_true if t in pred_text[1:]], i, color=(255,255,0))
            _, i = draw_text_list(im, [t for t in texts_true if t not in pred_text], i, color=(0,0,255))
        _, i = draw_text_list(im, pred_labels, i)

        zxy = Zt_images[frame:frame+1]
        imshowed.set_data(im[:,:,::-1])
        imscatter.set_offsets(zxy)

        for l, i in zip(lines_topk, i_topkmax):
            l.set_data([zxy[0, 0], Zt_text[i, 0]], [zxy[0, 1], Zt_text[i, 1]])
            l.set_alpha(sim[i])
        return [imshowed, imscatter] + lines_topk

    def init():
        return animate(0)

    ani = pltanim.FuncAnimation(
        fig, animate, tqdm.tqdm(np.arange(len(ims))[:50]), init_func=init, 
        interval=int(1000/fps), blit=True)

    f=os.path.join(out_dir, 'emb-motion.mp4')
    ani.save(f, writer=pltanim.FFMpegWriter(fps=fps))
    plt.close()


if __name__ == '__main__':
    import fire
    fire.Fire(draw)