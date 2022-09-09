from __future__ import annotations
# import heartrate; heartrate.trace(browser=True)
import functools
from typing import AsyncIterator, cast
import os
import time
import contextlib
import datetime
import tqdm

import cv2
import numpy as np




def nowstring():
    return datetime.datetime.now().strftime('%Y.%m.%d-%H.%M.%S')


class Context:
    def __init__(self, **kw):
        self.kw = kw

    def context(self):
        raise NotImplementedError
        yield
    async def acontext(self):
        raise NotImplementedError
        yield

    __context = __acontext = None
    def __enter__(self, *a, **kw):
        self.__context = contextlib.contextmanager(self.context)(*a, **kw, **self.kw)
        return cast(self.__class__, self.__context.__enter__())

    def __exit__(self, *a):
        if self.__context:
            return self.__context.__exit__(*a)
        self.__context=None

    def __aenter__(self, *a, **kw):
        self.__acontext = contextlib.asynccontextmanager(self.acontext)(*a, **kw, **self.kw)
        return cast(self.__class__, self.__acontext.__aenter__())

    def __aexit__(self, *a):
        if self.__acontext:
            return self.__acontext.__aexit__(*a)
        self.__acontext=None

    def write(self, id, data):
        raise NotImplementedError


class StreamReader(Context):
    def __init__(self, api, streams, recording_id=None, prefix=None, raw=False, raw_ts=False, progress=True, merged=False, **kw):
        super().__init__(streams=streams, **kw)
        self.api = api
        self.recording_id = recording_id
        self.prefix = prefix or (f'{recording_id}:' if recording_id else None)
        self.raw = raw
        self.raw_ts = raw_ts
        self.merged = merged
        self.progress = progress

    async def acontext(self, streams, fullspeed=None, last=None, replay_pull_timeout=5000) -> 'AsyncIterator[StreamReader]':
        self.replayer = self._replay_task = None
        rid = self.recording_id
        if rid:
            async with self.api.recordings.replay_connect(
                    rid, '+'.join(streams), 
                    fullspeed=fullspeed, 
                    prefix=f'{self.prefix}:'
            ) as self.replayer:
                self._replay_task = asyncio.create_task(self.watch_replay())
                streams = [f'{self.prefix}:{s}' for s in streams]
                async with self.api.data_pull_connect('+'.join(streams), last=last, timeout=replay_pull_timeout) as self.ws:
                    yield self
            if not self._replay_task.done():
                self._replay_task.cancel()
            else:
                self._replay_task.result()
            self.pbar = None
            return

        async with self.api.data_pull_connect('+'.join(streams), last=last) as self.ws:
            yield self
        self.pbar = None


    async def watch_replay(self, done=None):
        if self.replayer is not None:
            try:
                await self.replayer.done(done)
            finally:
                self.running = False

    async def __aiter__(self):
        self.running = True
        from ptgctl import holoframe
        from ptgctl.util import parse_epoch_time
        pbar = tqdm.tqdm()
        while self.running:
            pbar.set_description('getting data...')
            data = await self.ws.recv_data()
            pbar.set_description(f'got {len(data)}')
            if self.prefix:
                data = [(sid[len(self.prefix)+1:], t, x) for (sid, t, x) in data]
            if self.merged:
                yield holoframe.load_all(data)
                pbar.update()
            else:
                for sid, t, x in data:
                    yield (
                        (sid, t, x) if self.raw else 
                        (sid, t, holoframe.load(x)) if self.raw_ts else
                        (sid, parse_epoch_time(t), holoframe.load(x)))
                    pbar.update()


class StreamWriter(Context):
    def __init__(self, api, streams, writer=None, test=False, **kw):
        super().__init__(streams=streams, **kw)
        self.api = api
        self.test = test
        self.writer = writer

    async def acontext(self, streams):
        if self.test:
            yield self
            return
        if self.writer:
            yield self
            return
        async with self.api.data_push_connect('+'.join(streams), batch=True) as self.ws:
            yield self

    async def write(self, data, t=None):
        if self.test:
            print(data)
            return
        if self.writer:
            self.writer.write(data, t)
            return
        await self.ws.send_data(data)


class ImageOutput:#'avc1', 'mp4v', 
    prev_im = None
    t_video = 0
    def __init__(self, src, fps, cc='mp4v', cc_fallback='avc1', fixed_fps=False, show=None):
        self.src = src
        self.cc = cc
        self.cc_fallback = cc_fallback
        self.fps = fps
        self.fixed_fps = fixed_fps
        self._show = not src if show is None else show
        self.active = self.src or self._show

    def __enter__(self):
        self.prev_im = None
        return self

    def __exit__(self, *a):
        if self._w:
            self._w.release()
        self._w = None
        self.prev_im = None
        if self._show:
            cv2.destroyAllWindows()
    async def __aenter__(self): return self.__enter__()
    async def __aexit__(self, *a): return self.__exit__(*a)

    def output(self, im, t=None):
        if self.src:
            if self.fixed_fps and t is not None:
                self.write_video_fixed_fps(im, t)
            else:
                self.write_video(im)
        if self._show:
            self.show_video(im)

    _w = None
    def write_video(self, im):
        if not self._w:
            ccs = [self.cc, self.cc_fallback]
            for cc in ccs:
                os.makedirs(os.path.dirname(self.src), exist_ok=True)
                self._w = cv2.VideoWriter(
                    self.src, cv2.VideoWriter_fourcc(*cc),
                    self.fps, im.shape[:2][::-1], True)
                if self._w.isOpened():
                    break
                print(f"{cc} didn't work trying next...")
            else:
                raise RuntimeError(f"Video writer did not open - none worked: {ccs}")
        self._w.write(im)

    def write_video_fixed_fps(self, im, t):
        if self.prev_im is None:
            self.prev_im = im
            self.t_video = t

        while self.t_video < t:
            self.write_video(self.prev_im)
            self.t_video += 1./self.fps
        self.write_video(im)
        self.t_video += 1./self.fps
        self.prev_im = im

    def show_video(self, im):
        cv2.imshow('output', im)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            raise StopIteration


class VideoInput:
    def __init__(self, src, fps=None, size=None, give_time=True, bad_frames_count=True, include_bad_frame=False):
        self.src = src
        self.dest_fps = fps
        self.size = size
        self.bad_frames_count = bad_frames_count
        self.include_bad_frame = include_bad_frame
        self.give_time = give_time

    def __enter__(self):
        self.cap = cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.src}")
        self.src_fps = src_fps = cap.get(cv2.CAP_PROP_FPS)
        self.every = int(src_fps/(self.dest_fps or src_fps))
        size = self.size or (
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame = np.zeros(tuple(size)+(3,)).astype('uint8')

        self.total = total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"{total/src_fps:.1f} second video. {total} frames @ {self.src_fps} fps,",
              f"reducing to {self.dest_fps} fps" if self.dest_fps else '')
        self.pbar = tqdm.tqdm(total=int(total))

    def __exit__(self, *a):
        self.cap.release()

    #def read_all(self, limit=None):

    def read_all(self, limit=None):
        ims = []
        with self:
            for t, im in self:
                if limit and t > limit/self.dest_fps:
                    break
                ims.append(im)
        return np.stack(ims)

    def __iter__(self):
        i=0
        while not self.total or self.pbar.n < self.total:
            ret, im = self.cap.read()
            self.pbar.update()

            if self.bad_frames_count: i += 1

            if not ret:
                self.pbar.set_description(f"bad frame: {ret} {im}")
                if not self.include_bad_frame:
                    continue
                im = self.frame
            self.frame = im

            if not self.bad_frames_count: i += 1

            if i%self.every:
                continue
            if self.size:
                im = cv2.resize(im, self.size)

            t = i / self.src_fps
            self.pbar.set_description(f"t={t:.1f}s")
            yield t if self.give_time else i, im



def video_feed(src: str|int=0, fps=None, give_time=True, bad_frames_count=True, include_bad_frame=False):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {src}")
    src_fps = round(cap.get(cv2.CAP_PROP_FPS))
    fps = fps or src_fps
    every = int(round(src_fps/fps))

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    last_frame = np.zeros((height, width, 3)).astype('uint8')

    i = -1
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"{total/src_fps:.1f} second video. {total} frames @ {src_fps} fps, reducing to {fps} fps")
    pbar = tqdm.tqdm(total=int(total))
    while not total or pbar.n < total:
        ret, im = cap.read()
        pbar.update()

        if bad_frames_count: i += 1

        if not ret:
            pbar.set_description(f"bad frame: {ret} {im}")
            if not include_bad_frame:
                continue

        if not bad_frames_count: i += 1

        if i%every:
            continue
        t = i / src_fps
        pbar.set_description(f"t={t:.1f}s")
        if im is None:
            im = last_frame
        else:
            last_frame = im
        yield t if give_time else i, im
    cap.release()


def get_video_info(src):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, total



def maybe_profile(func, min_time=20):
    @functools.wraps(func)
    def inner(*a, profile=False, **kw):
        if not profile:
            return func(*a, **kw)
        from pyinstrument import Profiler
        p = Profiler()
        t0 = time.time()
        try:
            with p:
                return func(*a, **kw)
        finally:
            if time.time() - t0 > min_time:
                p.print()
    return inner




def draw_boxes(im, boxes, labels):
    for xy, label in zip(boxes, labels if labels is not None else ['']*len(boxes)):
        xy = list(map(int, xy))
        im = cv2.rectangle(im, xy[:2], xy[2:4], (0,255,0), 2)
        if label:
            im = cv2.putText(im, label, xy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return im


def draw_text_list(img, texts, i=-1, tl=(10, 50), scale=0.4, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX, 
            scale, color, thickness)
    return img, i
