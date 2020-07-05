import logging
import logging.config
import logging.handlers
import multiprocessing
import random
import threading

from io import StringIO

class LoggerQueue(object):
    
    def __init__(self, in_logger=None):
        self.queue = multiprocessing.Manager().Queue()
        self.thread = None
        self.in_logger = in_logger
    
    def __enter__(self):
        self.thread = threading.Thread(target=self._logger_thread, args=(self.queue, self.in_logger))
        self.thread.start()
        return self.queue
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.queue.put(None)
        self.thread.join()
        
    def _logger_thread(self, queue, logger):
        while True:
            record = queue.get()
            if record is None:
                break
            if logger is None:
                logger = logging.getLogger(record.name)
            logger.handle(record)

def worker(a1, a2, q):
    qh = logging.handlers.QueueHandler(q)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(qh)
    
    logger = logging.getLogger(__name__)
    logger.info('{} {}'.format(a1, a2))
    
    return a1 + a2

def create_console_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(module)s][%(filename)s][%(lineno)d][%(funcName)s][%(name)s][%(relativeCreated)d][%(processName)s][%(threadName)s][%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    return logger

def create_buffer_logger(buf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(module)s][%(filename)s][%(lineno)d][%(funcName)s][%(name)s][%(relativeCreated)d][%(processName)s][%(threadName)s][%(levelname)s] %(message)s')
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    
    buf = StringIO()
    logger = create_buffer_logger(buf)
    
    with LoggerQueue(logger) as q:
        args = [(i, i*i, q) for i in range(10)]
        with multiprocessing.Pool() as p:
            pool_return_rs = p.starmap(worker, args)
    
    print(buf.getvalue())