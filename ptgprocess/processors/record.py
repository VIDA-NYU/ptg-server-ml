from __future__ import annotations
import os
import glob
import tqdm
import asyncio
import fnmatch
import contextlib
import datetime

import numpy as np

from .core import Processor
from ..util import StreamReader, Context
from ..record import BaseWriter, RawWriter, VideoWriter, AudioWriter, JsonWriter, RawReader




class BaseRecorder(Processor):
    Writer = BaseWriter

    raw = False
    STORE_DIR = 'recordings'
    STREAMS: list|None = None
    RECORDING_SID = 'event:recording:id'

    def new_recording_id(self):
        return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    recording_id = None

    async def call_async(self, recording_id=None, *a, **kw):
        if recording_id:
            return await self._call_async(recording_id, *a, **kw)
        recording_id = self.api.recordings.current()
        if not recording_id:
            print("waiting for recording to be activated")
            self.recording_id = recording_id = await self._watch_recording_id(recording_id)
        print("Starting recording:", recording_id)
        return await self._call_async(recording_id, *a, **kw)

    async def _watch_recording_id(self, recording_id):
        async with self.api.data_pull_connect(self.RECORDING_SID) as ws:
            for sid, ts, data in (await ws.recv_data()):
                data = data.decode('utf-8')
                if data != recording_id:
                    return data

    def select_streams(self, streams=None):
        if not streams:
            streams = self.api.streams.ls()
            if self.STREAMS:
                streams = [
                    s for s in self.api.streams.ls()
                    if any(fnmatch.fnmatch(s, k) for k in self.STREAMS)
                ]
        elif isinstance(streams, str):
            streams = streams.split('+')
        return streams

    def match_streams(self, streams):
        if self.STREAMS:
            streams = [
                s for s in streams or []
                if any(fnmatch.fnmatch(s, k) for k in self.STREAMS)
            ]
        return streams

    async def _call_async(self, recording_id=None, streams=None, replay=None, fullspeed=None, progress=True, store_dir=None, write_errors='ignore', **kw):
        from ptgctl.util import parse_epoch_time
        from ptgctl import holoframe

        store_dir = os.path.join(store_dir or self.STORE_DIR, recording_id or self.new_recording_id())
        os.makedirs(store_dir, exist_ok=True)

        streams = self.select_streams(streams)

        raw = getattr(self.Writer, 'raw', False)
        raw_ts = getattr(self.Writer, 'raw_ts', False)


        reader = StreamReader(
                    self.api, streams+[self.RECORDING_SID], recording_id=replay,
                    progress=progress, fullspeed=fullspeed, alternate_reader=None,
                    ack=False, onebyone=True, latest=False, raw=True, raw_ts=True)

        writers = {}
        with contextlib.ExitStack() as stack:
            async with reader:
                async for sid, t, x in reader:
                    if sid == self.RECORDING_SID:
                        print('\n'*3, "stopping recording", recording_id, x, '\n------------', flush=True)
                        break

                    try:
                        if not (raw or raw_ts):        
                            t = parse_epoch_time(t)
                        if not raw:
                            x = holoframe.load(x)
                    except Exception as e:
                        print(f'{type(e).__name__}: {e} - {t} {x}')
                        continue

                    if sid not in writers:
                        writers[sid] = stack.enter_context(
                            self.Writer(sid, store_dir, sample=x, t_start=t, **kw))
                    
                    try:
                        writers[sid].write(x, t)
                    except Exception as e:
                        print("error writing", sid, t, e.__class__.__name__, e)


    def run_missing(self, *rec_ids, raw_dir, streams=None, confirm=False, overwrite=False, **kw):
        from ptgctl.util import parse_epoch_time
        from ptgctl import holoframe
        selected_streams = streams

        assert os.path.abspath(raw_dir) != os.path.abspath(self.STORE_DIR), "um whatchu doin"
        all_rec_ids = os.listdir(raw_dir)
        rec_ids = rec_ids or '*'
        rec_ids = [r  for  r in all_rec_ids if any(fnmatch.fnmatch(r, p) for p in rec_ids)]
        for rec_id in rec_ids:
            try:
                print()
                print(rec_id, '----------')
                # get missing streams
                src_dir = os.path.join(raw_dir, rec_id)
                out_dir = os.path.join(self.STORE_DIR, rec_id)
                if not os.path.isdir(src_dir):
                    print("no recording named", rec_id)
                os.makedirs(out_dir, exist_ok=True)
                src_streams = {os.path.splitext(f)[0]: f for f in os.listdir(src_dir)}
                out_streams = {os.path.splitext(f)[0] for f in os.listdir(out_dir)}
                missing = set(src_streams)
                if not overwrite:
                    missing -= out_streams
                if selected_streams:
                    missing = set(selected_streams)&missing
                streams = self.match_streams(list(missing))
                print("found existing", out_streams)
                print("found raw", src_streams)
                print(rec_id, "missing", missing)
                print("writer matched:", set(streams))
                if not streams:
                    print('nothing to do, moving on.')
                    continue
                if confirm:
                    c=input('continue? [Y/c/n] ').lower()
                    if 'c' in c:
                        continue
                    if 'n' in c:
                        return

                pbar = tqdm.tqdm(streams)
                for sid in pbar:
                    pbar.set_description(sid)
                    with RawReader(os.path.join(src_dir, src_streams[sid])) as reader:
                        raw = getattr(self.Writer, 'raw', False)
                        raw_ts = getattr(self.Writer, 'raw_ts', False)

                        it = iter(reader)
                        xx = next(it, None)
                        if xx is None:
                            return
                        t, x = xx
                        
                        try:
                            if not (raw or raw_ts):
                                t = parse_epoch_time(t)
                            if not raw:
                                x = holoframe.load(x)
                        except Exception as e:
                            print(f'{type(e).__class__}: {e} - {t} {x}')
                            continue

                        wkw = dict(self.get_writer_params(sid), **kw)
                        with self.Writer(sid, out_dir, sample=x, t_start=t, **wkw) as writer:
                            raw = getattr(writer, 'raw', False)
                            raw_ts = getattr(writer, 'raw_ts', False)

                            it = (x for xs in [[xx],it] for x in xs)
                            for t, x in it:
                                try:
                                    if not (raw or raw_ts):
                                        t = parse_epoch_time(t)
                                    if not raw:
                                        x = holoframe.load(x)
                                    writer.write(x, t)
                                except Exception as e:
                                    print("error writing", sid, t, e.__class__.__name__, e)
            except Exception as e:
                import traceback
                traceback.print_exc()

    WRITER_PARAMS = {}
    def get_writer_params(self, sid):
        return self.WRITER_PARAMS.get(sid) or {}


class RawRecorder(BaseRecorder):
    Writer = RawWriter
    ext = '.zip'


class VideoRecorder(BaseRecorder):
    Writer = VideoWriter
    STREAMS=['main', 'depthlt', 'gll', 'glf', 'grf', 'grr']
    ext = '.mp4'

class AudioRecorder(BaseRecorder):
    Writer = AudioWriter
    STREAMS = ['mic0']
    ext = '.mp3'

class JsonRecorder(BaseRecorder):
    Writer = JsonWriter
    ext = '.json'
    STREAMS = [
        'hand', 'eye', 'imuaccel', 'imugyro', 'imumag', 'yolo*', 'clip*',
        'gllCal', 'glfCal', 'grfCal', 'grrCal', 'depthltCal', 'detic:*',
        'reasoning*',
        'egovlp*',
        'pointcloud',
        'event*',
    ]
    WRITER_PARAMS = {
        'pointcloud': {'max_fps': 1/3}
    }


if __name__ == '__main__':
    import fire
    fire.Fire({
        'video': VideoRecorder,
        'audio': AudioRecorder,
        'json': JsonRecorder,
        'raw': RawRecorder,
    })
