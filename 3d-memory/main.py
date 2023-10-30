import os

m3d = __import__('3d_memory') # module starts with a number
# from 3d_memory import Memory3DApp
from frame_synchornize import FrameSyncApp

if __name__ == '__main__':
    import fire
    fire.Fire({
        'mem': m3d.Memory3DApp,
        'sync': FrameSyncApp,
    })