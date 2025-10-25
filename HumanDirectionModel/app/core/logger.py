import logging

def get_logger(name="app"):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        log.addHandler(ch)
    return log
