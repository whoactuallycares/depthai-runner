from functools import wraps
import threading
import logging

def ImportantThread(name: str = ""):
  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      threadName = func.__name__ if name == "" else name
      die = threading.Event()

      def runner():
        retval = True
        while not die.is_set() and retval:
          try:
            logging.info(f"Starting thread '{threadName}'")
            retval = func(*args, **kwargs)
          except Exception as e:
            logging.error(f" Thread '{threadName}' crashed with error: {e}")
        logging.warn(f"Thread '{threadName}' was killed")

      t = threading.Thread(target=runner)
      t.start()

      return die

    return wrapper
  return decorator
