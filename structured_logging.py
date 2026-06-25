"""
Structured JSON logging for pipeline runs (optional file sink).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("run_id", "command", "game_date"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, separators=(",", ":"))


def configure_structured_logging(
    *,
    level: int = logging.INFO,
    log_path: Path | None = None,
    run_id: str | None = None,
) -> None:
    """Attach JSON file handler when STRUCTURED_LOG=1 or log_path set."""
    path = log_path
    if path is None and os.getenv("STRUCTURED_LOG", "").lower() in ("1", "true", "yes"):
        path = DATA_DIR / "pipeline.jsonl"
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(JsonLogFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    if run_id:
        logging.LoggerAdapter(root, {"run_id": run_id})
