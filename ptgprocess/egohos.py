import os
import sys
import glob
import numpy as np
import torch
from torch import nn

import cv2
from PIL import Image, ImageOps

#detic_path = os.getenv('DETIC_PATH') or 'Detic'
#sys.path.insert(0,  detic_path)

from mmseg.apis import init_segmentor
from mmcv.parallel import collate, scatter
from mmseg.datasets.pipelines import Compose

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL_DIR = os.path.join(os.getenv("MODEL_DIR") or 'models', 'egohos')

class BaseEgoHos(nn.Module):
    def __init__(self, config, checkpoint=None, device=device):
        super().__init__()
        if not os.path.isfile(config):
            self.name = config
            config = os.path.join(MODEL_DIR, config, f'{config}.py')
            assert os.path.isfile(config), f'{self.name} not in {os.listdir(MODEL_DIR)}'
        else:
            self.name = os.path.basename(os.path.dirname(config))
        if not checkpoint:
            checkpoint = max(glob.glob(os.path.join(os.path.dirname(config), '*.pth')))
        
        #print('using device:', device)
        if device == 'cpu':
            import mmcv
            config = mmcv.Config.fromfile(config)
            config["norm_cfg"]["type"] = "BN"
            config["model"]["backbone"]["norm_cfg"]["type"] = "BN"
            config["model"]["decode_head"]["norm_cfg"]["type"] = "BN"
            config["model"]["auxiliary_head"]["norm_cfg"]["type"] = "BN"
        self.model = init_segmentor(config, checkpoint, device=device)
        self.preprocess = Compose(self.model.cfg.data.test.pipeline[1:])

        self.device = device
        self.is_cuda = device != 'cpu'
        self.palette = get_palette(None, self.model.CLASSES)
        self.classes = self.model.CLASSES
        self.addt_model=None

    def forward(self, img):
        data = {'img': img, 'img_shape': img.shape, 'ori_shape': img.shape, 'filename': '__.png', 'ori_filename': '__.png'}
        data = self.preprocess(data)
        
        # add additional segmentation maps
        addt = self.addt_model(img) if self.addt_model is not None else None
        if addt is not None:
            data['img'] = [
                    torch.cat([im] + [self.pad_resize(x, im) for x in xs[::-1]], dim=0)
                for im, xs in zip(data['img'], addt)
            ]

        data = collate([data], samples_per_gpu=1)
        if self.is_cuda:  # scatter to specified GPU
            data = scatter(data, [self.device])[0]
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        
        cfg = self.model.cfg
        data['img_metas'][0][0].update({k: '' for k in ['additional_channel', 'twohands_dir', 'cb_dir']})

        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        
        result = [x[None] for x in result]
        if addt is not None:
            result = [np.concatenate([x]+list(xs)) for x, *xs in zip(result, addt)]
        return result

    def pad_resize(self, aux, im):
        _, h, w = im.shape
        ha, wa = aux.shape
        hn, wn = (int(h/w*wa), wa) if ha/wa < h/w else (ha, int(ha/(h/w)))
        dh, dw = (hn-ha)/2, (wn-wa)/2
        
        aux = np.pad(aux, ((int(dh), int(dw)), (int(np.ceil(dh)), int(np.ceil(dw)))), "constant", constant_values=0)
        aux = cv2.resize(aux.astype(float), (w, h))[None]
        return torch.Tensor(aux)

    def show_result(self, *a, **kw):
        return self.model.show_result(*a, **kw)



class EgoHosHands(BaseEgoHos):
    def __init__(self, config='seg_twohands_ccda', **kw):
        super().__init__(config, **kw)

class EgoHosCB(BaseEgoHos):
    def __init__(self, config='twohands_to_cb_ccda', **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosHands()

class EgoHosObj1(BaseEgoHos):
    def __init__(self, config='twohands_cb_to_obj1_ccda', **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosCB()

class EgoHosObj2(BaseEgoHos):
    def __init__(self, config='twohands_cb_to_obj2_ccda', **kw):
        super().__init__(config, **kw)
        self.addt_model = EgoHosCB()


MODELS = {'hands': EgoHosHands, 'obj1': EgoHosObj1, 'obj2': EgoHosObj2, 'cb': EgoHosCB}

class EgoHos(nn.Module):
    def __new__(self, mode='obj2', *a, **kw):
        return MODELS[mode](*a, **kw)
  


def get_palette(palette, classes):
    if palette is None:
        state = np.random.get_state()
        np.random.seed(42)
        palette = np.random.randint(0, 255, size=(len(classes), 3))
        np.random.set_state(state)
    palette = np.array(palette)
    assert palette.shape == (len(classes), 3)
    return palette[:,::-1]


def draw_segs(im, result, palette, opacity=0.5):
    seg = result[0]
    assert 0 < opacity <= 1.0
    opacity = (seg[...,None]!=0)*opacity
    im = im * (1 - opacity) + palette[seg] * opacity
    return im.astype(np.uint8)


#from IPython import embed
def run(src, out_file=None, fps=10, show=None, palette=None, opacity=0.5, **kw):
    """Run multi-target tracker on a particular sequence.
    """
    from ptgprocess.util import VideoInput, VideoOutput, draw_boxes

    model = EgoHos(**kw)
    print(model.classes)
    print(model.model.cfg['additional_channel'])

    if out_file is True:
        out_file = f'egohos{model.name}_{os.path.basename(src)}'


    with VideoInput(src, fps) as vin, \
         VideoOutput(out_file, fps, show=show) as imout:
        for i, im in vin:
            result = model(im)[0]
            #print(im.shape)
            #print([x.shape for x in result])
            #embed()
            im2 = draw_segs(im, result, model.palette)
            imout.output(im2)

if __name__ == '__main__':
    import fire
    fire.Fire(run)

