"""Centralized logging configuration for the elote library."""

import logging
import sys
from typing import Union, Optional, TextIO

# The main logger for the elote library
# Users can configure this logger using standard logging methods
# or the helper functions below.
logger = logging.getLogger("elote")

# Add a NullHandler by default to prevent logs from propagating
# unless the user configures logging.
logger.addHandler(logging.NullHandler())

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def set_level(level: Union[int, str]) -> None:
    """Set the logging level for the elote logger.

    Args:
        level: The logging level (e.g., logging.DEBUG, 'INFO').
    """
    logger.setLevel(level)


def add_handler(handler: logging.Handler) -> None:
    """Add a logging handler to the elote logger.

    Args:
        handler: The logging handler to add.
    """
    # Remove the NullHandler if other handlers are added
    if isinstance(logger.handlers[0], logging.NullHandler):
        logger.removeHandler(logger.handlers[0])
    logger.addHandler(handler)


def basic_config(
    level: Union[int, str] = logging.WARNING, stream: Optional[TextIO] = None, format: str = DEFAULT_FORMAT
) -> None:
    """Configure basic logging for elote.

    Sets the level and adds a StreamHandler (defaults to stderr)
    with the specified format.

    Args:
        level: The minimum logging level to output.
        stream: The stream to log to (e.g., sys.stdout). Defaults to sys.stderr.
        format: The log message format string.
    """
    set_level(level)
    handler = logging.StreamHandler(stream or sys.stderr)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    add_handler(handler)
