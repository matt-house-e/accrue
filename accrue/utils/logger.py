"""
Logging utilities with colored console output and JSON structure support.

This module provides utilities for setting up application logging with:
- Colored console output for development
- JSON structured logging for production
- Configurable log levels and formats
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from typing import Any, Literal


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability in development."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Create a copy to avoid modifying the original record
        record_copy = logging.makeLogRecord(record.__dict__)

        # Add colors to level name
        color = self.COLORS.get(record_copy.levelname, "")
        record_copy.levelname = f"{color}{self.BOLD}{record_copy.levelname}{self.RESET}"

        # Add colors to logger name
        record_copy.name = f"\033[90m{record_copy.name}{self.RESET}"  # Gray

        return super().format(record_copy)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    format_type: Literal["console", "json"] = "console",
    include_timestamp: bool = True,
) -> None:
    """
    Set up application logging with colored console or JSON formatting.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_type: Either 'console' for colored output or 'json' for structured logging.
        include_timestamp: Whether to include timestamps in console format.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    # Set up formatter based on format type
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        # Console format with colors
        if include_timestamp:
            format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        else:
            format_string = "%(name)s | %(levelname)s | %(message)s"

        formatter = ColoredFormatter(
            fmt=format_string,
            datefmt="%H:%M:%S",  # Simpler time format for console
        )

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # Quiet down noisy third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: The name for the logger (typically __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)


def log_with_context(logger: logging.Logger, level: str, message: str, **context) -> None:
    """
    Log a message with additional context fields (useful for JSON logging).

    Args:
        logger: The logger instance.
        level: Log level (info, debug, warning, error, critical).
        message: The log message.
        **context: Additional context fields to include.
    """
    # For JSON logging, we can add context as extra fields
    record = logging.LogRecord(
        name=logger.name,
        level=getattr(logging, level.upper()),
        pathname="",
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.extra_fields = context

    # Use the logger's handle method to process the record
    if logger.isEnabledFor(record.levelno):
        logger.handle(record)
