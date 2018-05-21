import logging
import os
from functools import wraps
from timeit import default_timer as timer

def setup_logs(save_file):
  ## initialize logger
  logger = logging.getLogger("HomeCredit")
  logger.setLevel(logging.INFO)

  ## create the logging file handler
  fh = logging.FileHandler(save_file)

  ## create the logging console handler
  ch = logging.StreamHandler()

  ## format
  formatter = logging.Formatter("%(asctime)s - %(message)s")
  fh.setFormatter(formatter)
  ch.setFormatter(formatter)

  ## add handlers to logger object
  logger.addHandler(fh)
  logger.addHandler(ch)

  return logger

def logspeed(f):
  @wraps(f)
  def wrapper(*args, **kwargs):
    logger = logging.getLogger("HomeCredit")
    start = timer()
    result = f(*args, **kwargs)
    end = timer()
    logger.info(f'{f.__name__} - elapsed time: {end-start} seconds')
    return result
  return wrapper
