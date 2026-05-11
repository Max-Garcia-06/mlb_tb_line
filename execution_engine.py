from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DATA_DIR

LEDGER_PATH = Path(DATA_DIR) / "execution_ledger.json"
TICK_SIZE = 0.01


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_to_tick(x: float) -> float:
    x = round(round(x / TICK_SIZE) * TICK_SIZE, 2)
    return min(0.99, max(0.01, x))


@dataclass(frozen=True)
class LedgerKey:
    game_date: str
    ticker: str
    side: str


class ExecutionLedger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self._data = {"version": 1, "entries": []}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                self._data = json.loads(self.path.read_text())
        except Exception:
            self._data = {"version": 1, "entries": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def has(self, key: LedgerKey) -> bool:
        k = {"game_date": key.game_date, "ticker": key.ticker, "side": key.side}
        return k in (e.get("key") for e in self._data.get("entries", []))

    def add_attempt(
        self,
        key: LedgerKey,
        *,
        price: float,
        contracts: int,
        dollars: float,
        note: str = "",
        order_id: str = "",
        success: Optional[bool] = None,
    ) -> None:
        self._data.setdefault("entries", []).append(
            {
                "ts": _utc_now_iso(),
                "key": {"game_date": key.game_date, "ticker": key.ticker, "side": key.side},
                "price": price,
                "contracts": contracts,
                "dollars": dollars,
                "order_id": order_id,
                "success": success,
                "note": note,
            }
        )
        self._save()


def suggest_limit_price(
    *,
    side: str,
    bid: float,
    ask: float,
    model_fair: float,
    max_cross_spread: float = 0.06,
) -> float:
    side = (side or "").lower()
    if side not in {"yes", "no"}:
        raise ValueError(f"Invalid side: {side!r}")

    bid = float(bid)
    ask = float(ask)
    model_fair = float(model_fair)
    spread = max(0.0, ask - bid)
    mid = (ask + bid) / 2

    if spread <= max_cross_spread:
        px = min(ask, model_fair)
    else:
        px = min(max(bid + TICK_SIZE, mid - TICK_SIZE), model_fair)

    return _round_to_tick(px)

