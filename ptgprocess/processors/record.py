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

    def new_recording_id(self):
        return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

    recording_id = None

    async def call_async(self, recording_id=None, *a, **kw):
        done = asyncio.Event()
        if recording_id:
            return await self._call_async(done, recording_id, *a, **kw)
        recording_id = self.api.recordings.current()
        if not recording_id:
            print("waiting for recording to be activated")
            self.recording_id = recording_id = await self._watch_recording_id(done, recording_id)
        print("Starting recording:", recording_id)
        return await asyncio.gather(
            self._call_async(done, *a, recording_id=recording_id, **kw),
            self._watch_recording_id(done, recording_id)
        )

    async def _watch_recording_id(self, done, recording_id):
        loop = asyncio.get_event_loop()
        while not done.is_set():
            new_recording_id, _ = await asyncio.gather(
                loop.run_in_executor(None, self.api.recordings.current),
                asyncio.sleep(3)
            )
            self.recording_id = new_recording_id
            if new_recording_id != recording_id:
                # print(self.recording_id, '!=', recording_id)
                if done:
                    done.set()
                return new_recording_id

    async def _call_async(self, done, recording_id=None, streams=None, replay=None, fullspeed=None, progress=True, store_dir=None, **kw):
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
                    self.api, streams, recording_id=replay, 
                    progress=progress, fullspeed=fullspeed, 
                    raw=raw, raw_ts=raw_ts) as reader:
                async def _stream():
                    async for sid, t, x in reader:
                        if done is not None and done.is_set():
                            break
                        if recording_id and self.recording_id != recording_id:
                            # print(self.recording_id, '!=', recording_id)
                            break

                        if sid not in writers:
                            writers[sid] = stack.enter_context(
                                self.Writer(sid, store_dir, sample=x, t_start=t, **kw))

                        writers[sid].write(x, t)
                    done.set()

                await asyncio.gather(_stream(), reader.watch_replay(done))

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
