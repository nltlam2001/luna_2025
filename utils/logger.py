# utils/logger.py
import logging
import logging.handlers
import os
from datetime import datetime
import pytz


class TZFormatter(logging.Formatter):
    """
    Formatter hiển thị timestamp theo timezone (mặc định Asia/Ho_Chi_Minh)
    """
    def __init__(self, fmt=None, datefmt=None, tzname="Asia/Ho_Chi_Minh"):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tz = pytz.timezone(tzname)

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
    when: str = "midnight",
    backup_count: int = 14,
):
    """
    Setup logger với:
    - Console log
    - File log (rotate theo ngày)
    - Timezone Việt Nam
    """

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger  # tránh add handler trùng

    log_format = (
        "[%(asctime)s] "
        "[%(levelname)s] "
        "[%(name)s] "
        "%(message)s"
    )

    formatter = TZFormatter(fmt=log_format)

    # -------- Console handler --------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # -------- File handler (rotate) --------
    file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=file_path,
        when=when,
        backupCount=backup_count,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
