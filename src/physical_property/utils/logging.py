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
import contextlib, logging, os, re, sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Dict, Optional, Iterable

# stay quiet unless app configures handlers
logging.getLogger("physical_property").addHandler(logging.NullHandler())

# optional Loguru
try:
    from loguru import logger as _loguru
    _HAVE_LOGURU = True
except Exception:
    _HAVE_LOGURU = False

_CONFIGURED = False

class _Bound(logging.LoggerAdapter):
    def bind(self, **extra):  # prettier context
        merged = {**self.extra, **extra}
        return _Bound(self.logger, merged)
    def success(self, msg, *a, **k):  # Loguru-like
        self.info(msg, *a, **k)

def get_logger(name: str, **context) -> _Bound:
    return _Bound(logging.getLogger(name), context)

def _ensure_parent_dir(p: Path):
    with contextlib.suppress(Exception):
        Path(p).parent.mkdir(parents=True, exist_ok=True)

def _parse_bytes(s: str, default=10 * 1024 * 1024) -> int:
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?B)\s*$", s or "", re.I)
    if not m: return default
    n, u = float(m.group(1)), m.group(2).upper()
    return int(n * {"KB":1024,"MB":1024**2,"GB":1024**3}.get(u,1))

def setup_logging(
    *,
    level: Optional[str] = None,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "14 days",  # Loguru: days; stdlib: backupCount≈days
    quiet_modules: Optional[Dict[str, str]] = None,
    use_loguru: Optional[bool] = None,
    json: bool = False, backtrace: bool = False, diagnose: bool = False,  # Loguru-only
    enable_namespaces: Optional[Iterable[str]] = None,
) -> None:
    """App/CLI entry. Library code should NOT call this."""
    global _CONFIGURED
    if _CONFIGURED: return

    lvl = (level or os.getenv("PHYS_PROP_LOG_LEVEL") or "WARNING").upper()
    log_dir = log_dir or os.getenv("PHYS_PROP_LOG_DIR") or "logs"
    use_loguru = (use_loguru if use_loguru is not None
                  else os.getenv("PHYS_PROP_USE_LOGURU","0") == "1")
    json = json or (os.getenv("PHYS_PROP_LOG_JSON","0") == "1")
    backtrace = backtrace or (os.getenv("PHYS_PROP_LOG_BACKTRACE","0") == "1")
    diagnose = diagnose or (os.getenv("PHYS_PROP_LOG_DIAGNOSE","0") == "1")
    quiet_modules = quiet_modules or {}
    enable_namespaces = tuple(enable_namespaces or ("physical_property",))

    file_path = Path(log_file or Path(log_dir)/"physical_property_{time}.log".format(time=""))
    if log_file is None:  # add timestamp only for stdlib path
        file_path = Path(log_dir)/"physical_property_{:s}.log".format(__import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S"))
    _ensure_parent_dir(file_path)

    # quiet chatty deps (works in both modes)
    for prefix, qlvl in quiet_modules.items():
        logging.getLogger(prefix).setLevel(getattr(logging, qlvl.upper(), logging.WARNING))

    if use_loguru and _HAVE_LOGURU:
        # -------- Loguru sinks --------
        _loguru.remove()
        if not json:
            fmt = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} | {message}"
            _loguru.add(sys.stderr, level=lvl, format=fmt, backtrace=backtrace, diagnose=diagnose, enqueue=True)
            file_fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
            _loguru.add(file_path, level=lvl, format=file_fmt, rotation=rotation, retention=retention, enqueue=True,
                        backtrace=backtrace, diagnose=diagnose)
        else:
            _loguru.add(sys.stderr, level=lvl, serialize=True, backtrace=backtrace, diagnose=diagnose, enqueue=True)
            _loguru.add(file_path, level=lvl, serialize=True, rotation=rotation, retention=retention, enqueue=True,
                        backtrace=backtrace, diagnose=diagnose)
        # bridge stdlib -> Loguru
        class _Intercept(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                try: lg = _loguru.level(record.levelname).name
                except Exception: lg = record.levelno
                _loguru.opt(depth=6, exception=record.exc_info).log(lg, record.getMessage())
        logging.basicConfig(handlers=[_Intercept()], level=0)
        for ns in enable_namespaces:
            with contextlib.suppress(Exception): _loguru.enable(ns)
    else:
        # -------- Pure stdlib sinks --------
        root = logging.getLogger()
        root.setLevel(getattr(logging, lvl, logging.INFO))
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(getattr(logging, lvl, logging.INFO))
        fmt = "%(asctime)s %(levelname)s %(name)s:%(funcName)s:%(lineno)d %(message)s"
        sh.setFormatter(logging.Formatter(fmt))
        root.addHandler(sh)
        max_bytes = _parse_bytes(rotation)
        # crude retention: "14 days" -> 14 backups
        import re as _re
        m = _re.match(r"\s*(\d+)\s*day", retention, _re.I)
        fh = RotatingFileHandler(file_path, maxBytes=max_bytes, backupCount=int(m.group(1)) if m else 0)
        fh.setLevel(getattr(logging, lvl, logging.INFO))
        fh.setFormatter(logging.Formatter(fmt))
        root.addHandler(fh)

    _CONFIGURED = True
