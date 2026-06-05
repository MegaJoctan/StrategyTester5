__version__ = '2.2.4'
__author__  = 'Omega Joctan Msigwa.'

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone

try:
    import MetaTrader5 as _mt5
    MT5_AVAILABLE = True

except ImportError:
    raise RuntimeWarning(
        "MetaTrader5 API is not installed.\n"
        "If you are operating on a non-windows os, ignore this warning but if you are on windows, install it with: pip install strategytester5[mt5]"
    )

def log_date_suffix():
    return datetime.now(timezone.utc).strftime("%Y%m%d")

LOG_DATE = log_date_suffix()

class SimulatedTimeFormatter(logging.Formatter):
    def __init__(self, *args, time_provider=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_provider = time_provider  # function returning datetime

    def formatTime(self, record, datefmt=None):
        if self.time_provider:
            dt = self.time_provider()
        else:
            dt = datetime.fromtimestamp(record.created)

        if datefmt:
            return dt.strftime(datefmt)

        return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_logger(task_name: str, logfile: str, level=logging.INFO, time_provider=None):
    """
        Returns a logger
    """

    logger_name = f"{task_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = SimulatedTimeFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)s - %(funcName)10s() ] => %(message)s",
        time_provider=time_provider
    )

    file_handler = RotatingFileHandler(
        logfile,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


