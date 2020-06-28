# -*- coding: utf-8 -*-
import logging
import logging.handlers
import multiprocessing
import threading

import time
import random
from io import StringIO

class SingleQueueHandler(logging.handlers.QueueHandler):
    
    _instance_lock = threading.Lock()

    def __init__(self, queue):
        super().__init__(queue)

    def __new__(cls, *args):
        if not hasattr(SingleQueueHandler, "_instance"):
            with SingleQueueHandler._instance_lock:
                if not hasattr(SingleQueueHandler, "_instance"):
                    SingleQueueHandler._instance = object.__new__(cls)
        return SingleQueueHandler._instance

def get_queue_logger(logger_queue):
    qh = SingleQueueHandler(logger_queue)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(qh)
    return logger

def logger_thread(logger_queue, logger):
    if not logger:
        return
    logger.info('logger thread start')
    while True:
        record = logger_queue.get()
        if record is None:
            logger.info('logger thread end')
            break
        logger.handle(record)

def worker(p_a, logger_queue):
    logger = None
    if logger_queue is not None:
        logger = get_queue_logger(logger_queue)
        logger.info('worker start')
        
        time.sleep(random.randint(2,3))
        
    try:
        raise ValueError('Something error')
    except ValueError as e:
        if logger:
            logger.exception(e)
    if logger:
        logger.info('worker end')
    return p_a

def mt_process(in_logger):
    if in_logger:
        in_logger.info('mt_process start')
    
    logger_queue = multiprocessing.Manager().Queue()
    logger_process = threading.Thread(target=logger_thread, args=(logger_queue, in_logger))
    logger_process.start()
    
    #####
    t_pool = multiprocessing.Pool(processes=4)
    pool_return_rs = []
    for i in range(10):
        pool_return_rs.append(t_pool.apply_async(worker, (i, logger_queue)))
    t_pool.close()
    t_pool.join()
    #####
    
    logger_queue.put(None)
    logger_process.join()
    pool_return_rs = [e.get(1e6) for e in pool_return_rs]
    if in_logger:
        in_logger.info('mt_process end')
    return pool_return_rs

def create_console_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(module)s][%(filename)s][%(lineno)d][%(funcName)s][%(name)s][%(relativeCreated)d][%(threadName)s][%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    return logger

def create_buffer_logger(buf):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '[%(asctime)s][%(module)s][%(filename)s][%(lineno)d][%(funcName)s][%(name)s][%(relativeCreated)d][%(threadName)s][%(levelname)s] %(message)s')
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(file_formatter)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    
    buf = StringIO()
    
    #t_logger = create_console_logger()
    t_logger = create_buffer_logger(buf)
    mt_process(t_logger)
    
    #buf.seek(0)
    print(buf.getvalue())
    buf.close()
    