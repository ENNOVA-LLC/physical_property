""" `utils.logging`

Centralized logging utilities for the `physical_property` package.

Design goals
------------
- Library-friendly: no sinks/handlers are added on import.
- Configurable: one function (`setup_logging`) sets console/file sinks.
- Quiet-by-default: default level can be WARNING to avoid import-time noise.
- Composable: bridge stdlib logging to Loguru (so deps flow into our logs).
- Contextual: `get_logger(__name__, **context)` binds fields to every record.

Typical usage
-------------
# In an *application* / CLI that uses physical_property:
from physical_property.utils.logging import setup_logging
setup_logging(level="INFO", quiet_modules={"some.chatty.dep": "WARNING"})

# In *library modules* within physical_property:
from physical_property.utils.logging import get_logger
logger = get_logger(__name__)
logger.debug("Something helpful for debugging")

Environment variables (optional)
--------------------------------
PHYS_PROP_LOG_LEVEL      (default: "WARNING")
PHYS_PROP_LOG_DIR        (default: "logs")
PHYS_PROP_LOG_FILE       (default: none -> auto timestamped file under LOG_DIR)
PHYS_PROP_LOG_JSON       (default: "0")
PHYS_PROP_LOG_BACKTRACE  (default: "0")
PHYS_PROP_LOG_DIAGNOSE   (default: "0")
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from loguru import logger as _base_logger  # the global Loguru logger

# Idempotency guard so setup_logging() can be called multiple times safely.
_CONFIGURED = False


class InterceptHandler(logging.Handler):
    """Route stdlib logging records into Loguru, preserving levels and exceptions."""
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _base_logger.level(record.levelname).name
        except Exception:
            level = record.levelno
        # Depth: try to point to the original caller (not logging internals)
        _base_logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def _ensure_parent_dir(path: Path) -> None:
    with contextlib.suppress(Exception):
        path.parent.mkdir(parents=True, exist_ok=True)


def setup_logging(
    *,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    json: bool = False,
    backtrace: bool = False,
    diagnose: bool = False,
    rotation: str = "10 MB",
    retention: str = "14 days",
    compression: Optional[str] = None,  # e.g. "zip"
    quiet_modules: Optional[Dict[str, str]] = None,
) -> None:
    """Configure Loguru sinks for applications that use `physical_property`.

    Call this once at your app/CLI entry point. Library modules should *not*
    add sinks; they merely obtain a bound logger via `get_logger`.

    Parameters
    ----------
    level : str, optional
        Minimum level for console/file sinks (e.g., "INFO", "WARNING", "DEBUG").
        If omitted, uses env PHYS_PROP_LOG_LEVEL or defaults to "WARNING".
    log_dir : str, optional
        Directory for the log file. Defaults to env PHYS_PROP_LOG_DIR or "logs".
    log_file : str, optional
        Explicit log file path. If omitted, a timestamped file is created under `log_dir`.
    json : bool, default False
        Emit structured JSON logs (overrides text formatting).
        Can also be enabled via PHYS_PROP_LOG_JSON=1.
    backtrace : bool, default False
        Enable Loguru backtrace detail (env: PHYS_PROP_LOG_BACKTRACE=1).
    diagnose : bool, default False
        Enable Loguru diagnose mode (env: PHYS_PROP_LOG_DIAGNOSE=1).
    rotation : str, default "10 MB"
        Log rotation size/time (Loguru syntax).
    retention : str, default "14 days"
        Retention policy for rotated logs.
    compression : str, optional
        Compression for rotated logs (e.g., "zip").
    quiet_modules : dict[str, str], optional
        Mapping of module prefixes to a minimum level to allow, e.g.:
        {"physical_property": "WARNING", "urllib3": "WARNING"}.

    Notes
    -----
    This function is idempotent; subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Resolve configuration from env (with sensible defaults)
    level = (level or os.getenv("PHYS_PROP_LOG_LEVEL") or "WARNING").upper()
    log_dir = log_dir or os.getenv("PHYS_PROP_LOG_DIR") or "logs"
    json = json or (os.getenv("PHYS_PROP_LOG_JSON", "0") == "1")
    backtrace = backtrace or (os.getenv("PHYS_PROP_LOG_BACKTRACE", "0") == "1")
    diagnose = diagnose or (os.getenv("PHYS_PROP_LOG_DIAGNOSE", "0") == "1")
    quiet_modules = quiet_modules or {}

    # Remove any pre-existing default sink to prevent duplicate logs
    _base_logger.remove()

    # Filtering function to quiet specific module trees
    def _filter(record):
        name = record["name"]  # e.g., "physical_property.units.units"
        for prefix, min_level in quiet_modules.items():
            if name.startswith(prefix):
                return record["level"].no >= _base_logger.level(min_level).no
        return True

    # Console sink
    if not json:
        console_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> "
            "| {name}:{function}:{line} | {message}"
        )
        _base_logger.add(
            sys.stderr,
            level=level,
            filter=_filter,
            format=console_fmt,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=True,
        )
    else:
        _base_logger.add(
            sys.stderr,
            level=level,
            filter=_filter,
            serialize=True,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=True,
        )

    # File sink (optional but useful for apps)
    if log_file:
        file_path = Path(log_file)
    else:
        file_path = Path(log_dir) / "physical_property_{time:YYYYMMDD_HHmmss}.log"
    _ensure_parent_dir(file_path)

    if not json:
        file_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
        _base_logger.add(
            file_path,
            level=level,
            filter=_filter,
            format=file_fmt,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=True,
        )
    else:
        _base_logger.add(
            file_path,
            level=level,
            filter=_filter,
            serialize=True,
            rotation=rotation,
            retention=retention,
            compression=compression,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=True,
        )

    # Bridge stdlib logging -> Loguru, so third-party `logging` users flow into our sinks
    logging.basicConfig(handlers=[InterceptHandler()], level=0)

    _CONFIGURED = True


def get_logger(module_name: str, **context):
    """Return a Loguru logger bound with module and optional context fields.

    Example
    -------
    >>> from physical_property.utils.logging import get_logger
    >>> logger = get_logger(__name__, pkg="physical_property")
    >>> logger.debug("ready")
    """
    return _base_logger.bind(name=module_name, **context)
