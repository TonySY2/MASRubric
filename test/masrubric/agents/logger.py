import logging
from datetime import datetime

class TimeFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        s = dt.strftime("%H:%M:%S")
        ms = int(record.msecs)
        return f"{s}.{ms:03d}"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"

formatter = TimeFormatter(fmt=LOG_FORMAT)

handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.DEBUG)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False  
    return logger
