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

# Set a reasonable default level to avoid excessive debug logging
logger.setLevel(logging.WARNING)

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def set_level(level: Union[int, str]) -> None:
    """Set the logging level for the elote logger.

    Args:
        level: The logging level (e.g., logging.DEBUG, logging.INFO, 'DEBUG', 'INFO').
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


def add_handler(handler: logging.Handler) -> None:
    """Add a handler to the elote logger.

    Args:
        handler: A logging handler to add.
    """
    # Remove existing handlers of the same type to avoid duplicates
    for existing_handler in logger.handlers[:]:
        if isinstance(existing_handler, type(handler)):
            logger.removeHandler(existing_handler)
    
    logger.addHandler(handler)


def basic_config(
    level: Union[int, str] = logging.WARNING, 
    stream: Optional[TextIO] = None, 
    format: str = DEFAULT_FORMAT,
    force: bool = False
) -> None:
    """Configure basic logging for elote.

    Sets the level and adds a StreamHandler (defaults to stderr)
    with the specified format.

    Args:
        level: The minimum logging level to output.
        stream: The stream to log to (e.g., sys.stdout). Defaults to sys.stderr.
        format: The log message format string.
        force: If True, remove existing handlers before adding new one.
    """
    if force:
        # Remove all existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    set_level(level)
    handler = logging.StreamHandler(stream or sys.stderr)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    add_handler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Optional name for the logger. If None, returns the main elote logger.
        
    Returns:
        A logger instance.
    """
    if name is None:
        return logger
    return logging.getLogger(f"elote.{name}")


def disable_debug_logging() -> None:
    """Disable debug logging for performance in production environments."""
    if logger.level <= logging.DEBUG:
        logger.setLevel(logging.INFO)
        

def is_debug_enabled() -> bool:
    """Check if debug logging is enabled.
    
    Returns:
        True if debug logging is enabled, False otherwise.
    """
    return logger.isEnabledFor(logging.DEBUG)
