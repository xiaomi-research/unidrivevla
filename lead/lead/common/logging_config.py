"""
Central logging configuration for the LEAD project.

This module provides a single point to configure logging for all components.
1. Import and call setup_logging() at the start of your application.
2. Create a logger in each module using logging.getLogger(name)
"""

import logging
import os
import sys

from beartype import beartype


# ANSI color codes for terminal output
class ColorCodes:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Colors
    RED = "\033[31m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log levels and uses relative paths."""

    # Color mapping for different log levels
    LEVEL_COLORS = {
        logging.DEBUG: ColorCodes.BRIGHT_BLACK,
        logging.INFO: ColorCodes.BRIGHT_BLUE,
        logging.WARNING: ColorCodes.BRIGHT_YELLOW,
        logging.ERROR: ColorCodes.BRIGHT_RED,
        logging.CRITICAL: ColorCodes.BOLD + ColorCodes.RED,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get current working directory for relative path calculation
        self.cwd = os.getcwd()

    def format(self, record):
        # Save original values
        original_levelname = record.levelname
        original_pathname = record.pathname

        # Convert absolute path to relative path
        try:
            record.pathname = os.path.relpath(record.pathname, self.cwd)
        except ValueError:
            # If relpath fails (e.g., different drives on Windows), keep absolute
            pass

        # Add color to level name
        if record.levelno in self.LEVEL_COLORS:
            color = self.LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{record.levelname}{ColorCodes.RESET}"

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.pathname = original_pathname

        return result


@beartype
def setup_logging(level: str = None, format_string: str = None):
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Can be set via LEAD_LOG_LEVEL environment variable.
               Defaults to INFO.
        format_string: Custom format string for log messages.
                      Defaults to a standard format with timestamp.
    """
    # Determine log level
    if level is None:
        # Check environment variable first
        env_level = os.environ.get("LEAD_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)
    elif isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Default format with timestamp, level, file path, line number, and message
    if format_string is None:
        format_string = (
            "%(asctime)s [%(levelname)s] [%(pathname)s:%(lineno)d] %(message)s"
        )

    # Configure root logger with basic config first
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Get root logger
    logger = logging.getLogger()

    # Replace handler with colored formatter if outputting to terminal
    if sys.stdout.isatty():
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create new handler with colored formatter
        handler = logging.StreamHandler(sys.stdout)
        formatter = ColoredFormatter(fmt=format_string, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Suppress third-party loggers - only show warnings and errors
    for logger_name in [
        "matplotlib",
        "PIL",
        "urllib3",
        "asyncio",
        "srunner",
        "leaderboard",
        "agents",
        "carla",
        "timm",
    ]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set root logger to WARNING by default, so only lead.* modules show INFO
    logging.getLogger().setLevel(logging.INFO)

    # Enable your lead modules at the specified level
    logging.getLogger("lead").setLevel(level)

    logger.info(
        "Logging configured: level=%s for 'lead' modules",
        logging.getLevelName(level),
    )
