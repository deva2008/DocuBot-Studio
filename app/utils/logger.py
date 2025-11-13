import logging
import sys

_logger = None

def get_logger(name: str) -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    _logger = logger
    return logger
