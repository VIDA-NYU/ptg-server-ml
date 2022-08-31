from __future__ import annotations
import os
import asyncio
import orjson
import contextlib
import datetime

import numpy as np

from .core import Processor
from .util import Context, StreamReader


class BaseWriter(Context):
    def __init__(self, **kw):
        super().__init__(**kw)
    def context(self, sample, t_start): yield self
    def write(self, data, t): raise NotImplementedError


class RawWriter(BaseWriter):
    raw=True
    def __init__(self, name, store_dir, **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.zip')

    def context(self, sample, t_start):
        import zipfile
        print("Opening zip file:", self.fname)
        with zipfile.ZipFile(self.fname, 'w', zipfile.ZIP_STORED, False) as self.writer:
            yield self

    def write(self, data, ts):
        self.writer.writestr(ts, data)


class VideoWriter(BaseWriter):
    def __init__(self, name, store_dir, sample, t_start, fps=15, vcodec='libx264', crf='23',  **kw):
        super().__init__(**kw)
        fname = os.path.join(store_dir, f'{name}.mp4')
        
        self.prev_im = sample['image'][:,:,::-1].tobytes()
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

    def write(self, data, ts=None):
        im = data['image'][:,:,::-1].tobytes()
        if ts is not None:
            while self.t < ts - self.t_start:
                self.writer.write(self.prev_im)
                self.t += 1.0 / self.fps
            self.prev_im = im
        self.writer.write(im)
        self.t += 1.0 / self.fps


class AudioWriter(BaseWriter):
    def __init__(self, name, store_dir, **kw):
        self.fname = os.path.join(store_dir, f'{name}.wav')
        super().__init__(**kw)

    def context(self, sample, **kw):
        x = sample['audio']
        self.channels = x.shape[1] if x.ndim > 1 else 1
        self.lastpos = None
        import soundfile
        print("Opening audio file:", self.fname)
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
    def __init__(self, name, store_dir, **kw):
        super().__init__(**kw)
        self.fname = os.path.join(store_dir, f'{name}.json')
        
    def context(self, **kw):
        self.i = 0
        print("Opening json file:", self.fname)
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
            d['timestamp'] = ts
        self.fh.write(orjson.dumps(
            d, option=orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY))
        self.i += 1


class CsvWriter(BaseWriter):
    def __init__(self, name, store_dir, **kw):
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


class BaseRecorder(Processor):
    Writer = BaseWriter

    raw = False
    STORE_DIR = 'recordings'
    STREAMS: list|None = None

    def new_recording_id(self):
        return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    recording_id = None

    async def call_async(self, streams=None, recording_id=None, replay=None, fullspeed=None, progress=True, store_dir=None, **kw):
        store_dir = os.path.join(store_dir or self.STORE_DIR, recording_id or self.new_recording_id())
        os.makedirs(store_dir, exist_ok=True)

        if not streams:
            streams = self.api.streams.ls()
            if self.STREAMS:
                streams = [s for s in self.api.streams.ls() if any(s.endswith(k) for k in self.STREAMS)]
        elif isinstance(streams, str):
            streams = streams.split('+')

        raw = getattr(self.Writer, 'raw', False)
        raw_ts = getattr(self.Writer, 'raw_ts', False)

        writers = {}
        with contextlib.ExitStack() as stack:
            async with StreamReader(
                    self.api, streams, recording_id=replay, 
                    progress=progress, fullspeed=fullspeed, 
                    raw=raw, raw_ts=raw_ts) as reader:
                async def _stream():
                    async for sid, t, x in reader:
                        if recording_id and self.recording_id != recording_id:
                            break

                        if sid not in writers:
                            writers[sid] = stack.enter_context(
                                self.Writer(sid, store_dir, sample=x, t_start=t, **kw))

                        writers[sid].write(x, t)

                await asyncio.gather(_stream(), reader.watch_replay())

class RawRecorder(BaseRecorder):
    Writer = RawWriter
    


class VideoRecorder(BaseRecorder):
    Writer = VideoWriter


class AudioRecorder(BaseRecorder):
    Writer = AudioWriter


class JsonRecorder(BaseRecorder):
    Writer = JsonWriter



if __name__ == '__main__':
    import fire
    fire.Fire({
        'video': VideoRecorder,
        'audio': AudioRecorder,
        'json': JsonRecorder,
        'raw': RawRecorder,
    })
