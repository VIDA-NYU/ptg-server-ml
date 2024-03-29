from __future__ import annotations
import os
import glob
import tqdm
import orjson
import numpy as np
from .util import Context


class BaseWriter(Context):
    def __init__(self, **kw):
        super().__init__(**kw)
    def context(self, sample, t_start): yield self
    def write(self, data, t): raise NotImplementedError

MB = 1024 * 1024

class RawWriter(BaseWriter):
    raw=True
    def __init__(self, name, store_dir='', max_len=1000, max_size=9.5*MB, **kw):
        super().__init__(**kw)
        #self.fname = os.path.join(store_dir, f'{name}.zip')
        self.dir = os.path.join(store_dir, name)
        os.makedirs(self.dir, exist_ok=True)
        self.name = name
        self.max_len = max_len
        self.max_size = max_size

    def context(self, sample=None, t_start=None):
        try:
            self.size = 0
            self.buffer = []
            with tqdm.tqdm(total=self.max_len, desc=self.name) as self.pbar:
                yield self
        finally:
            if self.buffer:
                self._dump(self.buffer)
                self.buffer.clear()

    def _dump(self, data):
        if not data:
            return
        import zipfile
        fname = os.path.join(self.dir, f'{data[0][1]}_{data[-1][1]}.zip')
        tqdm.tqdm.write(f"writing {fname}")
        with zipfile.ZipFile(fname, 'a', zipfile.ZIP_STORED, False) as zf:
            for d, ts in data:
                zf.writestr(ts, d)

    def write(self, data, ts):
        self.pbar.update()
        self.size += len(data)
        self.buffer.append([data, ts])
        if len(self.buffer) >= self.max_len or self.size >= self.max_size:
            self._dump(self.buffer)
            self.buffer.clear()
            self.pbar.reset()
            self.size = 0


class VideoWriter(BaseWriter):
    def __init__(self, name, store_dir, sample, t_start, fps=15, vcodec='libx264', crf='23', scale=None, norm=None, max_duplicate_secs=10,  **kw):
        super().__init__(**kw)
        self.fname = fname = os.path.join(store_dir, f'{name}.mp4')

        if name == 'depthlt':
            scale = 40 if scale is None else scale
        self.scale = scale
        self.norm = norm
        
        self.prev_im = self.dump(sample['image'])
        self.t_start = t_start
        h, w = sample['image'].shape[:2]

        self.fps = fps
        self.max_duplicate = max_duplicate_secs * fps
        self.cmd = (
            f'ffmpeg -y -s {w}x{h} -pixel_format bgr24 -f rawvideo -r {fps} '
            f'-i pipe: -vcodec {vcodec} -pix_fmt yuv420p -crf {crf} {fname}')

    def context(self):
        import subprocess, shlex, sys
        process = subprocess.Popen(
            shlex.split(self.cmd), 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            #stderr=sys.stderr
        )
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
            print('finished', self.fname)

    def dump(self, im):
        if self.scale:
            im = (im * self.scale).astype(im.dtype)
        if self.norm:
            im = im / im.max()

        # convert int32 to uint8
        if not np.issubdtype(im.dtype, np.uint8):
            if np.issubdtype(im.dtype, np.integer):
                im = im.astype(float) / np.iinfo(im.dtype).max
            im = (im * 255).astype(np.uint8)
        if im.ndim == 2:
            im = np.broadcast_to(im[:,:,None], im.shape+(3,))
        return im[:,:,::-1].tobytes()

    def write(self, data, ts=None):
        im = self.dump(data['image'])
        if ts is not None:
            #i = 0
            while self.t < ts - self.t_start:
                self.writer.write(self.prev_im)
                self.t += 1.0 / self.fps
                #i += 1
                #if i > self.max_duplicate:
                #    self.t = ts
                #    break
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
    #raw=True
    raw_ts=True
    def __init__(self, name, store_dir='', max_fps=None, **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.json')
        self.max_fps = None
        
    def context(self, **kw):
        self.i = 0
        self.t_last = 0
        print("Opening json file:", self.fname, flush=True)
        with open(self.fname, 'wb') as self.fh:
            self.fh.write(b'[\n')
            try:
                yield self
            finally:
                self.fh.write(b'\n]\n')
                print('closing json', self.fname, flush=True)

    def write(self, d, ts=None):
        # frame dropping
        if ts is not None:
            t = int(ts.split('-')[0])/1000
            if self.max_fps and 1 / (t - self.t_last) > self.max_fps:
                return
            self.t_last = t
            
        if self.i:
            self.fh.write(b',\n')

        if ts is not None:
            try:
                if isinstance(d, bytes):
                    d = orjson.loads(d)
                if not isinstance(d, dict):
                    d = {'data': d}
                d['timestamp'] = ts
            except Exception:
                print("error reading data")
                print(d)
                raise
        try:
            if not isinstance(d, bytes):
                d = orjson.dumps(d, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
        except Exception:
            print("error writing data")
            print(d)
            raise
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


class ParquetWriter(BaseWriter):
    raw=True
    def __init__(self, name, store_dir='', **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.parquet')

    def context(self, sample, t_start=None):
        print("Opening csv file:", self.fname)
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.read_table = lambda d: pq.read_table(pa.BufferReader(d))

        table = self.read_table(sample)
        writer = pq.ParquetWriter(self.fname, table.schema)
        try:
            self._w = writer
            yield self
        finally:
            writer.close()
            self.on_close(self.fname)

    def write(self, data, ts):
        self._w.write_table(self.read_table(data))

    def format(self, x):
        if isinstance(x, float):
            return float(f'{x:.4f}')
        return x

    def on_close(self, fname):
        return 



class PointCloudWriter(ParquetWriter):
    def on_close(self, fname):
        from ptgprocess.voxelize import run
        out_fname = os.path.join(os.path.dirname(fname), "voxelized-pointcloud.json")
        run(fname, out_fname)


class RawReader:
    reader = None
    def __init__(self, src):
        if os.path.isdir(src):
            fs = sorted(glob.glob(os.path.join(src, '*')))
        else:
            if not os.path.isfile(src) and os.path.isfile(f'{src}.zip'):
                src = f'{src}.zip'
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
                fs = sorted(self.reader.namelist())
                pbar = tqdm.tqdm(fs)
                for ts in pbar:
                    pbar.set_description(f'{ts}')
                    with self.reader.open(ts, 'r') as f:
                        yield ts, f.read()
            self.reader = None

    def describe(self):
        from ptgctl.util import parse_epoch_time
        t_last = None
        for ts, d in self:
            t = parse_epoch_time(ts)
            tqdm.tqdm.write(f'{ts} {t - (t_last or t):.3f}s {len(d)}')
            t_last = t


if __name__ == '__main__':
    import fire
    fire.Fire()
