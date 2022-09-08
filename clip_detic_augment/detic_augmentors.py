import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch

import tqdm

import torchvision.transforms.functional as F


from PIL import Image, ImageOps, ImageFilter

class GroupRandomCrop(object):
    def __init__(self, size):
        self.size = (int(size), int(size)) if isinstance(size, numbers.Number) else size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size

        out_images = []
        for img in img_group:
            assert img.size == (w, h)
            if not (w == tw and h == th):
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                img = img.crop((x1, y1, x1 + tw, y1 + th))
            out_images.append(img)
        return out_images


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        self.scale_worker = GroupScale(scale_size) if scale_size is not None else None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)
        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        oversample_group = list()
        for o_w, o_h in GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h):
            crops = [img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h)) for img in img_group]
            flip_group = []
            for i, crop in enumerate(crops):
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)
                if crop.mode == 'L' and i % 2 == 0:
                    flip_crop = ImageOps.invert(flip_crop)
                flip_group.append(flip_crop)

            oversample_group.extend(crops)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupFCSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        self.scale_worker = GroupScale(scale_size) if scale_size is not None else None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)
        image_w, image_h = img_group[0].size

        oversample_group = []
        for o_w, o_h in GroupMultiScaleCrop.fill_fc_fix_offset(image_w, image_h, image_h, image_h):
            oversample_group.extend([img.crop((o_w, o_h, o_w + image_h, o_h + image_h)) for img in img_group])
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True, choice=None):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR
        self.choice = choice # 0 to 5 or 13 depending on more_fix_crop

    def __call__(self, img_group):
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(img_group[0].size)
        return [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
               .resize(self.input_size[:2], self.interpolation)
            for img in img_group
        ]

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[:2]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_choices = [
            (w, h) 
            for i, h in enumerate([self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]) 
            for j, w in enumerate([self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]) 
            if abs(i - j) <= self.max_distort
        ]
        crop_w, crop_h = random.choice(crop_choices) if self.choice is None else crop_choices[self.choice]
        w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_w, crop_h)
        return crop_w, crop_h, w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        if not self.fix_crop:
            return random.randint(0, image_w - crop_w), random.randint(0, image_h - crop_h)
        return random.choice(self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h))

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w)
        h_step = (image_h - crop_h)
        offsets = [(0,0), (1,0), (0,1), (1,1), (0.5,0.5)]
        ret = [(w*w_step, h*h_step) for w,h in offsets]

        if more_fix_crop:
            offsets = [(0,0.5), (1,0.5), (0.5,1), (0.5,0), (0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)]
            ret.extend([(w*w_step, h*h_step) for w,h in offsets])
        return np.array(ret, dtype=int)

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w)
        h_step = (image_h - crop_h)
        offsets = [(0,0), (0.5,0.5), (1,1)]
        return np.array([(w*w_step, h*h_step) for w,h in offsets], dtype=int)


class GroupRandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.scale = GroupScale(self.size, interpolation=self.interpolation)
        self.crop = GroupRandomCrop(self.size)

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                out_group = []
                for img in img_group:
                    img = img.crop((x1, y1, x1 + w, y1 + h))
                    assert(img.size == (w, h))
                    out_group.append(img.resize((self.size, self.size), self.interpolation))
                return out_group

        return self.crop(self.scale(img_group))


fixed = lambda x: (lambda: x)
uniform = lambda min=0, max=1, size=None: (lambda: np.random.uniform(min, max, size=size))

class randomwalk:
    def __init__(self, min=0.0, max=1.0, step=0.1, init=None, size=None):
        self.last = np.random.random(size) * (max-min) + min if init is None else np.asarray(init)
        self.step = np.asarray(step)
        self.min = np.asarray(min)
        self.max = np.asarray(max)
        self.size = size

    def __str__(self):
        return f'{self.__class__.__name__}({self.last}, min={self.min}, max={self.max}, step={self.step})'

    def __call__(self):
        last = self.last + self.step * (np.random.random(self.size) - 0.5)
        last = last - np.maximum(last - self.max, 0)*2 - np.minimum(last - self.min, 0)*2
        self.last = last
        return last

class RandomGroup:
    p = None
    def __init__(self, p=None, mix=None):
        if p is not None:
            self.p = p
        self.mix = fixed(mix) if isinstance(mix, float) else mix

    def __call__(self, img_group):
        if self.p is None or random.random() < self.p:
            return self.apply_workers(img_group)
        return img_group

    def apply_workers(self, img_group):
        return [self.mix_worker(img) for img in img_group]

    def mix_worker(self, img):
        if self.mix is None:
            return self.worker(img)
        return Image.blend(img, self.worker(img), self.mix())

    def worker(self, img):
        raise NotImplementedError

class GroupSolarization(RandomGroup):
    def worker(self, img):
        return ImageOps.solarize(img)

class GroupGaussianBlur(RandomGroup):
    def apply_workers(self, img_group):
        sigma = random.random() * 1.9 + 0.1
        return [img.filter(ImageFilter.GaussianBlur(sigma))  for img in img_group]

class GroupRandomGrayscale(RandomGroup):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.grey_worker = torchvision.transforms.Grayscale(num_output_channels=3)

    def worker(self, img):
        return self.grey_worker(img)

class GroupRandomColorJitter(RandomGroup):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, **kw):
        super().__init__(**kw)
        self.worker = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue)

class GroupRandomColorAlter(RandomGroup):
    def __init__(self, 
            brightness_factor=None, 
            contrast_factor=None,
            gamma_factor=None,
            hue_factor=None,
            saturation_factor=None,
            sharpness_factor=None,
            posterize_factor=None,
            solarize_factor=None, **kw):
        super().__init__(**kw)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.gamma_factor = gamma_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor
        self.sharpness_factor = sharpness_factor
        self.posterize_factor = posterize_factor
        self.solarize_factor = solarize_factor
        print({k: v.last for k, v in self.__dict__.items() if isinstance(v, randomwalk)})

    def worker(self, im):
        if self.brightness_factor:
            im = F.adjust_brightness(im, self.brightness_factor())
        if self.contrast_factor:
            im = F.adjust_contrast(im, self.contrast_factor())
        if self.gamma_factor:
            im = F.adjust_gamma(im, self.gamma_factor())
        if self.hue_factor:
            im = F.adjust_hue(im, self.hue_factor())
        if self.saturation_factor:
            im = F.adjust_saturation(im, self.saturation_factor())
        if self.sharpness_factor:
            im = F.adjust_sharpness(im, self.sharpness_factor())
        if self.posterize_factor:
            im = F.posterize(im, int(self.posterize_factor()))
        if self.solarize_factor:
            # im = F.solarize(im, self.solarize_factor())
            im = ImageOps.solarize(im, 255*self.solarize_factor())
        return im


class GroupCenterCrop(RandomGroup):
    def __init__(self, size, p=None):
        super().__init__(p)
        self.worker = torchvision.transforms.CenterCrop(size)
    
class GroupRandomHorizontalFlip(RandomGroup):
    def worker(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    

class GroupScale(RandomGroup):
    def __init__(self, size, interpolation=Image.BICUBIC):
        super().__init__()
        self.worker = torchvision.transforms.Resize(size, interpolation)


class GroupNormalize1(RandomGroup):
    def __init__(self, mean, std, p=None):
        super().__init__(p)
        self.worker = torchvision.transforms.Normalize(mean,std)
        
class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.Tensor(self.mean * (tensor.size()[0]//len(self.mean)))
        std = torch.Tensor(self.std * (tensor.size()[0]//len(self.std)))

        if len(tensor.size()) == 3: # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4: # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor


class Stack1:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if self.roll:
            return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
        return torch.from_numpy(np.concatenate(img_group, axis=0))

class Stack(Stack1):
    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return super().__call__([np.expand_dims(x, 2) for x in img_group])
        elif img_group[0].mode == 'RGB':
            return super().__call__(img_group)

class ToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):  # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        elif isinstance(pic, Image.Image):  # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        else:
            assert isinstance(pic, torch.Tensor)
            img = pic
        return img.float().div(255) if self.div else img.float()

class ToTorchFormatTensor1(RandomGroup):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, p=None):
        super().__init__(p)
        self.worker = torchvision.transforms.ToTensor()

class IdentityTransform(object):
    def __call__(self, data):
        return data


class Thumbnail(RandomGroup):
    def __init__(self, size, **kw):
        super().__init__(**kw)
        self.size = size
    def worker(self, im):
        return im.thumbnail(self.size, Image.ANTIALIAS)


def _perspective_coeffs(pa, pb):
    A = np.matrix([
        p for p1, p2 in zip(pa, pb)
        for p in (
            [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]],
            [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]]
        )
    ], dtype=float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

PREF = np.asarray([(0, 0), (1, 0), (1, 1), (0, 1)])
def persp_coeffs(im, pa, pb=PREF):
    return _perspective_coeffs(
        np.asarray(pa) * np.array(im.size), 
        np.asarray(pb) * np.array(im.size))

class GroupPerspective(RandomGroup):
    def __init__(self, distortion, **kw):
        super().__init__(**kw)
        self.distortion = distortion
        print(self.distortion())

    def worker(self, im):
        pa = self.distortion()
        pa /= np.linalg.norm(pa)
        pa = PREF + pa * (PREF * 2 - 1)
        return im.transform(im.size, Image.PERSPECTIVE, persp_coeffs(im, pa))
        


class ToPIL:
    def __call__(self, img_group):
        if isinstance(img_group, np.ndarray):
            return [Image.fromarray(img_group)]
        return img_group


def resize_augmentation(input_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        Thumbnail(600),
    ])

def get_augmentation(training, input_size, crop_choice=None):
    import random
    import torchvision
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = input_size * 256 // 224
    return torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Lambda(lambda x: [x]),
        *(
            [
                GroupScale((400, 600)),
                # GroupPerspective(randomwalk(
                #     np.array([(0,0), (0,0), (0.2,0.2), (0.2,0.2)]),
                #     np.array([(0,0), (0,0), (0.9,0.5), (0.9,0.5)]), 
                #     0.05, (4, 2))),
                GroupPerspective(randomwalk(0, 0.9, 0.025, size=(4, 2))),
                # GroupPerspective(lambda: np.array([(0,0), (0,0), (0.9,0.0), (0.9,0.0)])),
                # GroupMultiScaleCrop(input_size, [1, .875, .75, .66], choice=crop_choice),
                GroupRandomHorizontalFlip(p=bool(random.random() > 0.5)),
                # GroupRandomColorJitter(mix=randomwalk(0, 0.2, 0.01), brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                # GroupRandomGrayscale(mix=randomwalk(0, 0.4, 0.05)),
                # GroupGaussianBlur(mix=randomwalk(0, 1, 0.1)),
                # GroupSolarization(mix=randomwalk(0, 0.5, 0.05))
                GroupRandomColorAlter(
                    brightness_factor=randomwalk(0.7, 1.5, 0.05), 
                    contrast_factor=randomwalk(0.5, 1.5, 0.05),
                    gamma_factor=randomwalk(0.5, 2, 0.05),
                    hue_factor=randomwalk(0.09, 0.1, 0.001),
                    saturation_factor=randomwalk(0.5, 1.5, 0.05),
                    sharpness_factor=randomwalk(0.5, 1.5, 0.05),
                    # posterize_factor=randomwalk(5, 8, 0.3),
                    # solarize_factor=randomwalk(0.7, 1, 0.05)
                ),
            ]
            if training else
            [
                GroupScale(scale_size),  
                GroupCenterCrop(input_size)
            ]
        ), 
        Stack(roll=False),
        # torchvision.transforms.PILToTensor(),
        # GroupNormalize(input_mean, input_std)
    ])


def test_it(src, training=True, input_size=600, fps=10, out_file=None, show=None, loop=None, accel=1, **kw):
    from ptgprocess.util import VideoInput, VideoOutput

    if out_file is True:
        out_file = 'augment.mp4'

    with VideoOutput(out_file, fps*accel, show=show) as vout:
        while True:
            augs = get_augmentation(training, input_size, **kw)
            with VideoInput(src, fps) as vin:
                for t, im in vin:
                    if loop and t > loop:
                        break
                    im = augs(im).numpy()
                    vout.output(im, t)
            if not loop:
                break

def test_randomwalk():
    import time
    # walk = randomwalk()
    # while True:
    #     print(walk())
    #     time.sleep(0.2)
    walk = randomwalk(0, 0.5)
    for i in range(10):
        print(walk)
        print(walk())
        time.sleep(0.2)

    walk = randomwalk(0, 0.5, init=np.random.random((3,))*0.25, size=3)
    for i in range(10):
        print(walk)
        print(walk())
        time.sleep(0.2)

    walk = randomwalk(
        np.zeros(3), np.arange(3)+1, 
        init=np.arange(3)+1 + 0.5, size=3)
    for i in range(10):
        print(walk)
        print(walk())
        time.sleep(0.2)


if __name__ == '__main__':
    import fire
    fire.Fire({'walk': test_randomwalk, 'test': test_it})