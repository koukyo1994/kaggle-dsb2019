import logging
import time

from contextlib import contextmanager


@contextmanager
def timer(name: str, log: bool = False):
    t0 = time.time()
    msg = f"[{name}] start"
    if not log:
        print(msg)
    else:
        logging.info(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2d} s"
    if not log:
        print(msg)
    else:
        logging.info(msg)
