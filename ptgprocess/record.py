from __future__ import annotations
import os
import orjson
import numpy as np
from .util import Context


class BaseWriter(Context):
    def __init__(self, **kw):
        super().__init__(**kw)
    def context(self, sample, t_start): yield self
    def write(self, data, t): raise NotImplementedError


class RawWriter(BaseWriter):
    raw=True
    def __init__(self, name, store_dir='', **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.zip')

    def context(self, sample=None, t_start=None):
        import zipfile
        print("Opening zip file:", self.fname)
        with zipfile.ZipFile(self.fname, 'a', zipfile.ZIP_STORED, False) as self.writer:
            yield self

    def write(self, data, ts):
        self.writer.writestr(ts, data)


class VideoWriter(BaseWriter):
    def __init__(self, name, store_dir, sample, t_start, fps=15, vcodec='libx264', crf='23',  **kw):
        super().__init__(**kw)
        fname = os.path.join(store_dir, f'{name}.mp4')
        
        self.prev_im = self.dump(sample['image'])
        self.t_start = t_start
        h, w = sample['image'].shape[:2]

        self.fps = fps
        self.cmd = (
            f'ffmpeg -y -s {w}x{h} -pixel_format bgr24 -f rawvideo -r {fps} '
            f'-i pipe: -vcodec {vcodec} -pix_fmt yuv420p -crf {crf} {fname}')

    def context(self):
        import subprocess, shlex, sys
        process = subprocess.Popen(
            shlex.split(self.cmd), 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=sys.stderr)
        self.writer = process.stdin

        self.t = 0
        try:
            print("Opening video ffmpeg process:", self.cmd)
            yield self
        except BrokenPipeError as e:
            print(f"Broken pipe writing video: {e}")
            if process.stderr:
                print(process.stderr.read())
            raise e
        finally:
            print('finishing')
            if process.stdin:
                process.stdin.close()
            process.wait()
            print('finished')

    def dump(self, im):
        if im.ndim == 2:
            im = np.broadcast_to(im[:,:,None], im.shape+(3,))
        return im[:,:,::-1].tobytes()

    def write(self, data, ts=None):
        im = self.dump(data['image'])
        if ts is not None:
            while self.t < ts - self.t_start:
                self.writer.write(self.prev_im)
                self.t += 1.0 / self.fps
            self.prev_im = im
        self.writer.write(im)
        self.t += 1.0 / self.fps


class AudioWriter(BaseWriter):
    def __init__(self, name, store_dir='', **kw):
        self.fname = os.path.join(store_dir, f'{name}.wav')
        super().__init__(**kw)

    def context(self, sample, **kw):
        x = sample['audio']
        self.channels = x.shape[1] if x.ndim > 1 else 1
        self.lastpos = None
        import soundfile
        print("Opening audio file:", self.fname, flush=True)
        with soundfile.SoundFile(self.fname, 'w', samplerate=sample['sr'], channels=self.channels) as self.sf:
            yield self

    def write(self, d, t=None):
        pos = d['pos']
        y = d['audio']
        if self.lastpos:
            n_gap = min(max(0, pos - self.lastpos), d['sr'] * 2)
            if n_gap:
                self.sf.write(np.zeros((n_gap, self.channels)))
        self.lastpos = pos + len(y)


class JsonWriter(BaseWriter):
    raw=True
    def __init__(self, name, store_dir='', **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.json')
        
    def context(self, **kw):
        self.i = 0
        print("Opening json file:", self.fname, flush=True)
        with open(self.fname, 'wb') as self.fh:
            self.fh.write(b'[\n')
            try:
                yield self
            finally:
                self.fh.write(b'\n]\n')

    def write(self, d, ts=None):
        if self.i:
            self.fh.write(b',\n')
        if ts is not None:
            if isinstance(d, bytes):
                d = orjson.loads(d)
            if not isinstance(d, dict):
                d = {'data': d}
            d['timestamp'] = ts
        if not isinstance(d, bytes):
            d = orjson.dumps(d, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
        self.fh.write(d)
        self.i += 1


class CsvWriter(BaseWriter):
    def __init__(self, name, store_dir='', **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.csv')

    def context(self, header):
        print("Opening csv file:", self.fname)
        import csv
        with open(self.fname, 'w') as f:
            self._w = csv.writer(f)
            self._w.writerow(header)
            yield self

    def write(self, row):
        self._w.writerow([self.format(x) for x in row])

    def format(self, x):
        if isinstance(x, float):
            return float(f'{x:.4f}')
        return x


class RawReader:
    def __init__(self, src):
        if os.path.isdir(src):
            fs = sorted(glob.glob(os.path.join(src, '*')))
        else:
            fs = [src]
        self.fs = fs

    def __enter__(self):
        return self
    def __exit__(self, *a):
        if self.reader is not None:
            return self.reader.__exit__(*a)

    def __iter__(self):
        import zipfile
        for f in self.fs:
            with zipfile.ZipFile(f, 'r', zipfile.ZIP_STORED, False) as self.reader:
                for ts in sorted(self.reader.namelist()):
                    with zf.open(ts, 'r') as f:
                        yield ts, f.read()
            self.reader = None


