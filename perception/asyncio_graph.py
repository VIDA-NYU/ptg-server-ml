import asyncio
import traceback
import collections


class Graph:
    def __init__(self):
        self.producers = []
        self.queues = []
        self.consumers = []

    def add_producer(self, func, *queues, **kw):
        queues = [q or self.add_queue() for q in queues or (None,)]
        self.producers.append(asyncio.create_task(error_handler(func, *queues, **kw)))
        return queues
    
    def add_queue(self, cls=asyncio.Queue, *a, **kw):
        queue = cls(*a, **kw)
        self.queues.append(queue)
        return queue

    def add_consumer(self, func, queue, output_queue=None, **kw):
        a = (queue, output_queue,) if output_queue else (queue,)
        self.consumers.append(asyncio.create_task(error_handler(
            func, *a, **kw)))
        return output_queue
    
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return await self.run()
    
    async def run(self):
        try:
            # with both producers and consumers running, wait for
            # the producers to finish
            await asyncio.gather(*self.producers)
        finally:
            print("closing...")
            # wait for the remaining tasks to be processed
            print('closing queues...')
            print([q._unfinished_tasks for q in self.queues])
            await asyncio.gather(*(q.join() for q in self.queues))
            # cancel the consumers, which are now idle
            print('closing consumers...')
            for c in self.consumers:
                c.cancel()
            print('closed')


async def error_handler(func, *a, __retry_every=5, **kw):
    while True:
        try:
            return await func(*a, **kw)
        except Exception:
            traceback.print_exc()
            if __retry_every:
                await asyncio.sleep(__retry_every)
                continue
            raise


class SlidingQueue(asyncio.Queue):
    def __init__(self, maxsize=1, buffersize=2):
        self.buffersize = buffersize
        super().__init__(maxsize)

    def _init(self, maxsize):
        self._queue = collections.deque(maxlen=maxsize)
        self._buffer = collections.deque(maxlen=self.buffersize)

    def _put(self, item):
        self._queue.append(item)
        self._buffer.append(item)

    def read_buffer(self):
        xs = list(self._buffer)
        self._buffer.clear()
        return xs
    
    def push(self, item):
        full = self.full()
        self._put(item)
        self._unfinished_tasks += not full
        self._finished.clear()
        self._wakeup_next(self._getters)



async def sample_producer(q, sleep=1, limit=20, name='put'):
    print('starting', name)
    i = 0
    while True:
        # await q.put(i)
        q.push(i)
        print(name, i, q.qsize())
        await asyncio.sleep(sleep)
        i += 1
        if i > limit: 
            break

async def sample_consumer(q, q2=None, sleep=0.1, name='get'):
    print('starting', name)
    while True:
        print(name, 'before get')
        i = await q.get()
        print(name, 'after get')
        try:
            xs = q.read_buffer() if hasattr(q, 'read_buffer') else [i]
            print(name, i, xs, q.qsize())
            if q2 is not None:
                await q2.put(f'aaaa{i}')
            await asyncio.sleep(sleep)
        finally:
            q.task_done()
    print('done', name)


async def main():
    async with Graph() as g:
        q, = g.add_producer(sample_producer, g.add_queue(SlidingQueue, 1, 6), sleep=0.5)
        q2 = g.add_consumer(sample_consumer, q, g.add_queue(), sleep=3)
        g.add_consumer(sample_consumer, q2, name='get2')
        # g.add_consumer(sample_consumer, q, sleep=3, name='get1')
        # g.add_consumer(sample_consumer, q, sleep=3, name='get2')
        # g.add_consumer(sample_consumer, q, sleep=3, name='get3')

if __name__ == '__main__':
    asyncio.run(main())