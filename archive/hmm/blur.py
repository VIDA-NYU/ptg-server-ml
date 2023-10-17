import os
import cv2


def run(src, out_file=None, fps=30, R=11, limit=None, **kw):
    from ptgprocess.egohos import EgoHos, merge_segs
    #from ptgprocess.egovlp import EgoVLP
    from ptgprocess.util import VideoInput, VideoOutput

    if out_file is True:
        out_file = f'blurred_{os.path.basename(src)}'

    #model = EgoVLP()
    hosmodel = EgoHos(mode='objs')
    print(hosmodel.classes)

    

    with VideoInput(src, fps) as vin, VideoOutput(out_file, fps=fps) as vout:
        for t, im in vin:
            if limit and t > limit:
                break
            result = hosmodel(im)[0]
            mask = (result.sum(0) > 0)[...,None]
            blurred = cv2.GaussianBlur(im, (R, R), cv2.BORDER_DEFAULT)
            im_out = im * mask + blurred * (1 - mask)
            vout.output(im_out.astype('uint8'))

if __name__ == '__main__':
    import fire
    fire.Fire(run)
