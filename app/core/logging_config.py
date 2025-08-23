# app/logging_config.py
from __future__ import annotations
import logging
import sys

# Idempotent logger factory that plays well with Uvicorn/Starlette
def get_logger(name: str = "chat-app") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:   # already configured
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s:%(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Reduce noisy third‑party loggers
    for n in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "urllib3"):
        logging.getLogger(n).setLevel(logging.WARNING)

    # Don’t let child loggers duplicate to root
    logger.propagate = False
    return logger
