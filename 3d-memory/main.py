import os
import hand_detector


# Download the weights automatically from google drive
# this could be integrated into the hand_detector script 

def gdrive_download(path, file_id, base_url='https://docs.google.com/uc?export=download&confirm=t&id={}'):
    '''Guess you don't need gdown anymore!'''
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        pbar = lambda i, s, t: print(f'downloading checkpoint to {path}: {i * s / t:.2%}', end="\r")
        urllib.request.urlretrieve(base_url.format(file_id), path, pbar)
    return path

MODEL_PATH = "hand_object_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth"
FILE_ID = '1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE'

# patch in a wrapper class
# auto download the weights when loading the hand detector model if they aren't already
class HandDetector(hand_detector.HandDetector):
    def __init__(self, *a, **kw):
        gdrive_download(MODEL_PATH, FILE_ID)
        super().__init__(*a, **kw)
hand_detector.HandDetector = HandDetector



# bring up the CLI


m3d = __import__('3d_memory') # module starts with a number
# from 3d_memory import Memory3DApp
from frame_synchornize import FrameSyncApp

if __name__ == '__main__':
    import fire
    fire.Fire({
        'mem': m3d.Memory3DApp,
        'sync': FrameSyncApp,
    })