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


def draw_embedding_space(Zt, texts, Zt_path, ax=None):
    if ax:
        plt.sca(ax)
    else:
        ax = plt.gca()
    
    plt.scatter(Zt[:,0], Zt[:,1], label='text', c='r', s=50)

    ax.set_xticks([])
    ax.set_yticks([])
    text_annots = [ax.annotate(t, (z[0], z[1])) for z, t in zip(Zt, texts)]
    # adjust_text(
    #     text_annots, 
    #     arrowprops=dict(arrowstyle="-", color='b', lw=1), 
    #     only_move={'points':'y', 'text':'y'}, 
    #     lim=50)
    if Zt_path is not None:
        plt.plot(Zt_path[:,0], Zt_path[:,1])
        # plt.scatter(Zt_path[:,0], Zt_path[:,1])
    

def get_video(video_fname, fps, limit, size=(760, 428)):
    ims = []
    for t, im in video_feed(video_fname, fps=fps):
        if limit and t > limit/fps:
            break
        ims.append(cv2.resize(im, size))
    return np.stack(ims)


def draw(
    video_fname, model='action2', fps=10, topk=5,
    text_col='narration', normalized=True, test=False,
    recipe=None, recipe_key='steps', contrast_everything=False,
    checkpoint=None, ann_root=ANN_ROOT
):
    model = MODELS[model](checkpoint)
    
    df = pd.concat([
        pd.read_csv(os.path.join(ann_root, f"EPIC_100_train{'_normalized' if normalized else ''}.csv")).assign(split='train'),
        pd.read_csv(os.path.join(ann_root, f"EPIC_100_validation{'_normalized' if normalized else ''}.csv")).assign(split='val'),
    ])
    print(df.columns)
    print(df.shape)
    print(df.head())


    video_name = os.path.splitext(os.path.basename(video_fname))[0]
    

    if recipe:
        import ptgctl
        api = ptgctl.API()
        texts = api.recipes.get(recipe)[recipe_key]
        has_gt=False
        text_col = recipe_key
    else:
        df_vid = df[df.video_id == video_name]
        texts = (df if contrast_everything else df_vid)[text_col].unique()
        has_gt = True

    out_dir = f'output/{video_name}-{model.__class__.__name__}-{text_col}'
    os.makedirs(out_dir, exist_ok=True)

    print(f"Vocab ({len(texts)} words):", texts)

    Z_text = model.encode_text(texts).detach().numpy()

    # dimreduc = LocallyLinearEmbedding(n_components=2)
    dimreduc = LocallyLinearEmbedding(n_components=2, n_neighbors=20, method='hessian')
    Zt_text = dimreduc.fit_transform(Z_text)

    plt.figure(figsize=(16, 10))
    draw_embedding_space(Zt_text, texts, None)
    plt.savefig(os.path.join(out_dir, f'{text_col}-text-space.jpg'))
    plt.close()

    test = 50 if test is True else test if test else None
    ims = get_video(video_fname, fps, test)


    # make animated plot

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    ax1.axis('off')
    plt.tight_layout()

    imshowed = ax1.imshow(ims[0])

    draw_embedding_space(Zt_text, texts, None, ax=ax2)
    imscatter = ax2.scatter([], [], s=100)
    lines_topk = [ax2.plot([], [], c='r')[0] for i in range(topk)]
    imhistory, = ax2.plot([], [], alpha=0.15, c='k')
    imdothistory = ax2.scatter([], [], alpha=0.15, c='k')

    Zt_images = []

    def animate(frame):
        im = ims[frame].copy()

        Z_image = model.encode_image(im).detach()
        zxy = dimreduc.transform(Z_image)
        Zt_images.extend(zxy)

        similarity = torch.Tensor(100 * Z_image @ Z_text.T).softmax(dim=-1)[0]
        i_topkmax = torch.topk(similarity, topk, dim=-1)[1].detach().cpu().numpy().astype(int)
        sim = similarity.detach().tolist()
        
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

        
        imshowed.set_data(im[:,:,::-1])
        imscatter.set_offsets(zxy)
        imhistory.set_data([z[0] for z in Zt_images], [z[1] for z in Zt_images])
        imdothistory.set_offsets(Zt_images)

        for l, i in zip(lines_topk, i_topkmax):
            l.set_data([zxy[0, 0], Zt_text[i, 0]], [zxy[0, 1], Zt_text[i, 1]])
            l.set_alpha(sim[i])
        return [imshowed, imscatter, imhistory, imdothistory] + lines_topk

    ani = pltanim.FuncAnimation(
        fig, animate, tqdm.tqdm(np.arange(len(ims))[:test]),
        interval=int(1000/fps), blit=True)

    f=os.path.join(out_dir, 'emb-motion.mp4')
    ani.save(f, writer=pltanim.FFMpegWriter(fps=fps))
    plt.close()

    Zt_images = np.array(Zt_images)

    plt.figure(figsize=(16, 10))
    draw_embedding_space(Zt_text, texts, Zt_images)
    plt.scatter(Zt_images[:,0], Zt_images[:,1])
    plt.savefig(os.path.join(out_dir, 'text+image-space.jpg'))
    plt.close()



if __name__ == '__main__':
    import fire
    fire.Fire(draw)