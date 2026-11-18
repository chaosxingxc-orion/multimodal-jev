"""Bounded CPU preparation in original order; policy binding stays on consumer.

Use independent/thread-local processors in the worker callback. The callback
must never call distributed collectives, touch NPU, or bind a changing policy.
The context manager joins submitted work on close; it cannot interrupt Python
callbacks already executing. I/O callbacks must therefore have bounded waits.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class OrderedPrefetch:
    def __init__(self, prepare, items, *, workers=2, ahead=2):
        if type(workers) is not int or workers < 1 or type(ahead) is not int or ahead < 1:
            raise ValueError('positive workers and bounded lookahead required')
        self.prepare = prepare
        self.items = iter(items)
        self.ahead = ahead
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='jev-input')
        self.pending = deque()
        self.closed = False

    def _fill(self):
        while len(self.pending) < self.ahead:
            try: item = next(self.items)
            except StopIteration: break
            self.pending.append(self.executor.submit(self.prepare, item))

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed: raise StopIteration
        try:
            self._fill()
            if not self.pending:
                self.close()
                raise StopIteration
            result = self.pending.popleft().result()
            self._fill()  # prepare ahead while consumer trains this batch
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            for future in self.pending: future.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.pending.clear()

    def __exit__(self, *args):
        self.close()
