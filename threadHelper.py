from functools import wraps
import threading
import logging

def ImportantThread(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    die = threading.Event()

    def runner():
      while not die.is_set():
        try:
          logging.info(f"Starting thread '{func.__name__}'")
          func(*args, **kwargs)
        except Exception as e:
          logging.error(f" Thread '{func.__name__}' crashed with error: {e}")
      logging.warn(f"Thread '{func.__name__}' was killed")

    t = threading.Thread(target=runner)
    t.start()

    return die

  return wrapper
