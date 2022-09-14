from __future__ import annotations
import os
import asyncio
import fnmatch
import contextlib
import datetime

import numpy as np

from .core import Processor
from ..util import StreamReader
from ..record import BaseWriter, RawWriter, VideoWriter, AudioWriter, JsonWriter




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

    async def _call_async(self, recording_id=None, streams=None, replay=None, fullspeed=None, progress=True, store_dir=None, **kw):
        store_dir = os.path.join(store_dir or self.STORE_DIR, recording_id or self.new_recording_id())
        os.makedirs(store_dir, exist_ok=True)

        if not streams:
            streams = self.api.streams.ls()
            if self.STREAMS:
                streams = [
                    s for s in self.api.streams.ls() 
                    if any(fnmatch.fnmatch(s, k) for k in self.STREAMS)
                ]
        elif isinstance(streams, str):
            streams = streams.split('+')

        raw = getattr(self.Writer, 'raw', False)
        raw_ts = getattr(self.Writer, 'raw_ts', False)

        writers = {}
        with contextlib.ExitStack() as stack:
            async with StreamReader(
                    self.api, streams+[self.RECORDING_SID], recording_id=replay, 
                    progress=progress, fullspeed=fullspeed, 
                    ack=False, raw=raw, raw_ts=raw_ts) as reader:
                async for sid, t, x in reader:
                    if sid == self.RECORDING_SID:
                        break

                    if sid not in writers:
                        writers[sid] = stack.enter_context(
                            self.Writer(sid, store_dir, sample=x, t_start=t, **kw))

                    writers[sid].write(x, t)


class RawRecorder(BaseRecorder):
    Writer = RawWriter
    


class VideoRecorder(BaseRecorder):
    Writer = VideoWriter
    STREAMS=['main', 'depthlt', 'gll', 'glf', 'grf', 'grr']

class AudioRecorder(BaseRecorder):
    Writer = AudioWriter
    STREAMS = ['mic0']


class JsonRecorder(BaseRecorder):
    Writer = JsonWriter
    STREAMS = [
        'hand', 'eye', 'imuaccel', 'imugyro', 'imumag', 'yolo*', 'clip*',
        'gllCal', 'glfCal', 'grfCal', 'grrCal', 'depthltCal', 'detic:*',
    ]


if __name__ == '__main__':
    import fire
    fire.Fire({
        'video': VideoRecorder,
        'audio': AudioRecorder,
        'json': JsonRecorder,
        'raw': RawRecorder,
    })
