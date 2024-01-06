import os
import sys
import logging

def get_logger(path, fname):
    if not os.path.exists(path):
        os.mkdir(path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(
        os.path.join(path, fname), mode="a"
    )  # 'a' for append
    stderr_log_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(file_log_handler)
    logger.addHandler(stderr_log_handler)
    formatter = logging.Formatter(
        "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_log_handler.setFormatter(formatter)
    stderr_log_handler.setFormatter(formatter)
    sys.stdout.flush()

    return logger