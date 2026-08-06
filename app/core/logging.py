"""Structured JSON logging configuration."""

import logging
import sys
from typing import Any

from pythonjsonlogger.json import JsonFormatter


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with structured JSON output to stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove any pre-existing handlers to avoid duplicate output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
    )
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger — use this instead of logging.getLogger() directly."""
    return logging.getLogger(name)
