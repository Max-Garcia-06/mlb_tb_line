"""
kalshi_bridge.py (MLB TB)
------------------------
Kalshi API client with RSA request signing (v2 auth) and a mock layer.

This is adapted from the NBA rebound repo.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import (
    KALSHI_API_KEY_ID,
    KALSHI_BASE_URL,
    KALSHI_ORDER_URL,
    KALSHI_PRIVATE_KEY_PATH,
    REQUIRE_KALSHI_CREDENTIALS,
)
from flow_monitor import vpin_from_signed_volumes

log = logging.getLogger(__name__)


def read_price_from_market_dict(m: dict, dollars_key: str, cents_key: str) -> float:
    """Read a Kalshi price field as dollars (supports *_dollars or cent integers)."""
    if m.get(dollars_key) is not None:
        try:
            v = float(m[dollars_key])
            return v if v > 0 else 0.0
        except (TypeError, ValueError):
            pass
    if m.get(cents_key) is not None:
        try:
            return float(m[cents_key]) / 100.0
        except (TypeError, ValueError):
            pass
    return 0.0


@dataclass
class MarketLine:
    ticker: str
    player_name: str
    player_id: int
    game_date: str
    line: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    volume: int = 0
    open_interest: int = 0
    xref_player_id: str = ""
    vpin_toxicity: float = 0.0
    event_ticker: str = ""

    @property
    def yes_mid(self) -> float:
        return round((self.yes_ask + self.yes_bid) / 2, 4)

    @property
    def yes_spread(self) -> float:
        return round(self.yes_ask - self.yes_bid, 4)

    @property
    def no_spread(self) -> float:
        return round(self.no_ask - self.no_bid, 4)

    @property
    def implied_prob(self) -> float:
        return self.yes_mid


@dataclass
class OrderResult:
    success: bool
    order_id: str
    ticker: str
    side: str
    contracts: int
    price: float
    message: str = ""


@dataclass
class OpenOrder:
    order_id: str
    ticker: str
    side: str
    action: str
    type: str
    status: str
    price: float
    count: int
    remaining_count: int
    created_time: str = ""


def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign(private_key, timestamp_ms: int, method: str, path: str) -> str:
    message = f"{timestamp_ms}{method.upper()}/trade-api/v2{path}".encode("utf-8")
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


class KalshiClient:
    def __init__(
        self,
        key_id: str = KALSHI_API_KEY_ID,
        private_key_path: str = KALSHI_PRIVATE_KEY_PATH,
        base_url: str = KALSHI_BASE_URL,
        order_url: str = KALSHI_ORDER_URL,
    ):
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self.order_url = order_url.rstrip("/")
        self._private_key = _load_private_key(private_key_path)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _auth_headers(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": _sign(self._private_key, ts, method, path),
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.get(url, params=params, headers=self._auth_headers("GET", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.post(url, json=body, headers=self._auth_headers("POST", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.delete(url, params=params, headers=self._auth_headers("DELETE", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_markets(
        self,
        series_ticker: str,
        status: str = "open",
        limit: int = 200,
        max_pages: int = 20,
    ) -> list[dict]:
        markets: list[dict] = []
        cursor: Optional[str] = None
        pages = 0
        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params=params)
            batch = data.get("markets", []) or []
            markets.extend(batch)
            cursor = data.get("cursor")
            pages += 1
            if not cursor or not batch or pages >= max_pages:
                break
        return markets

    def get_total_bases_lines(self, game_date: Optional[str] = None, series_ticker: str = "KXMLBTB") -> list[MarketLine]:
        raw = self.get_markets(series_ticker=series_ticker)
        lines = self.parse_total_bases_markets(raw)
        if not game_date:
            return lines
        return [ml for ml in lines if ml.game_date == game_date]

    @staticmethod
    def _xref_from_market(m: dict) -> str:
        for k in (
            "xref_player_id",
            "xref_id",
            "cross_ref_player_id",
            "mlb_player_id",
            "player_xref_id",
            "custom_strike",
        ):
            v = m.get(k)
            if v is None:
                continue
            if isinstance(v, dict):
                for sub in ("player_id", "id", "xref", "mlb_id"):
                    if v.get(sub) is not None:
                        return str(v.get(sub)).strip()
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    def parse_total_bases_markets(self, raw_markets: list[dict]) -> list[MarketLine]:
        lines: list[MarketLine] = []
        for m in raw_markets:
            title = m.get("title", "") or ""
            if "total base" not in title.lower():
                continue
            ticker = m.get("ticker", "") or ""
            try:
                line = float(m["floor_strike"]) if m.get("floor_strike") is not None else self._extract_line_from_title(title)
                player_name = self._extract_player_from_title(title)
                xref = self._xref_from_market(m)

                def _price(dollars_key, cents_key, fallback):
                    v = read_price_from_market_dict(m, dollars_key, cents_key)
                    return v if v > 0 else fallback

                yes_ask = _price("yes_ask_dollars", "yes_ask", 0.50)
                yes_bid = _price("yes_bid_dollars", "yes_bid", 0.48)
                no_ask = _price("no_ask_dollars", "no_ask", 0.52)
                no_bid = _price("no_bid_dollars", "no_bid", 0.50)

                if yes_ask >= 0.99 and yes_bid <= 0.01:
                    continue

                event_ticker = m.get("event_ticker", "") or ""
                game_date_str = self._parse_game_date(event_ticker)
                lines.append(
                    MarketLine(
                        ticker=ticker,
                        player_name=player_name,
                        player_id=0,
                        game_date=game_date_str,
                        event_ticker=event_ticker,
                        line=line,
                        yes_ask=yes_ask,
                        yes_bid=yes_bid,
                        no_ask=no_ask,
                        no_bid=no_bid,
                        volume=int(m.get("volume", 0) or 0),
                        open_interest=int(m.get("open_interest", 0) or 0),
                        xref_player_id=xref,
                        vpin_toxicity=0.0,
                    )
                )
            except Exception:
                continue
        return lines

    @staticmethod
    def _parse_game_date(event_ticker: str) -> str:
        match = re.search(r"(\d{2})([A-Z]{3})(\d{2})", event_ticker)
        if match:
            year, mon, day = match.groups()
            months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                      "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
            return f"20{year}-{months.get(mon, '00')}-{day}"
        return datetime.today().strftime("%Y-%m-%d")

    @staticmethod
    def _extract_line_from_title(title: str) -> float:
        # "Player Name: 2+ total bases" -> line 1.5
        match = re.search(r":?\s*(\d+\.?\d*)\+?\s*total\s*bases?", title, re.IGNORECASE)
        if match:
            return float(match.group(1)) - 0.5
        match = re.search(r"(\d+\.5|\d+)", title)
        if match:
            return float(match.group(1))
        raise ValueError(f"Cannot parse line from: {title}")

    @staticmethod
    def _extract_player_from_title(title: str) -> str:
        match = re.match(r"^([^:]+):", title)
        if match:
            return match.group(1).strip().title()
        return title.strip().title()

    def _post_order(self, path: str, body: dict) -> dict:
        url = f"{self.order_url}{path}"
        r = self._session.post(url, json=body, headers=self._auth_headers("POST", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def place_order(self, ticker: str, side: str, contracts: int, price: float, order_type: str = "limit") -> OrderResult:
        side_norm = (side or "").strip().lower()
        body = {"ticker": ticker, "action": "buy", "side": side_norm, "count": contracts, "type": order_type}
        if side_norm == "yes":
            body["yes_price"] = int(round(price * 100))
        elif side_norm == "no":
            body["no_price"] = int(round(price * 100))
        else:
            return OrderResult(False, "", ticker, side, contracts, price, f"Invalid side: {side!r}")
        try:
            data = self._post_order("/portfolio/orders", body)
            order = data.get("order", {})
            return OrderResult(True, order.get("order_id", ""), ticker, side_norm, contracts, price, "Order placed")
        except requests.HTTPError as e:
            return OrderResult(False, "", ticker, side_norm, contracts, price, str(e))

    def get_orders(self, status: str = "resting", ticker: Optional[str] = None, limit: int = 200) -> list[OpenOrder]:
        params: dict = {"status": status, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        data = self._get("/portfolio/orders", params=params)
        orders = []
        for o in data.get("orders", []) or []:
            side = (o.get("side") or "").lower()
            price = None
            if o.get("yes_price_dollars") is not None and side == "yes":
                price = float(o["yes_price_dollars"])
            elif o.get("no_price_dollars") is not None and side == "no":
                price = float(o["no_price_dollars"])
            elif o.get("yes_price") is not None and side == "yes":
                price = float(o["yes_price"]) / 100
            elif o.get("no_price") is not None and side == "no":
                price = float(o["no_price"]) / 100
            elif o.get("price") is not None:
                price = float(o["price"])
            if price is None:
                price = 0.0
            orders.append(
                OpenOrder(
                    order_id=str(o.get("order_id", "")),
                    ticker=str(o.get("ticker", "")),
                    side=side,
                    action=str(o.get("action", "")),
                    type=str(o.get("type", "")),
                    status=str(o.get("status", "")),
                    price=float(price),
                    count=int(o.get("count", 0) or 0),
                    remaining_count=int(o.get("remaining_count", o.get("count", 0)) or 0),
                    created_time=str(o.get("created_time", "")),
                )
            )
        return orders

    def get_order(self, order_id: str) -> dict:
        """
        Fetch a single order record. We keep this as a raw dict because Kalshi's schema
        can change and different environments may expose different fields.
        """
        data = self._get(f"/portfolio/orders/{order_id}")
        return data.get("order", data)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._delete(f"/portfolio/orders/{order_id}")
            return True
        except requests.HTTPError:
            return False

    def get_market(self, ticker: str) -> dict:
        data = self._get(f"/markets/{ticker}")
        return data.get("market", data)

    def get_trade_signed_sizes(self, ticker: str, limit: int = 200) -> list[float]:
        """
        Best-effort signed trade sizes (+ buy, - sell) for VPIN. Returns [] if endpoint unavailable.
        """
        try:
            data = self._get("/markets/trades", params={"ticker": ticker, "limit": int(limit)})
        except Exception:
            return []
        out: list[float] = []
        for t in data.get("trades", []) or []:
            try:
                cnt = float(t.get("count") or t.get("quantity") or 0)
            except (TypeError, ValueError):
                cnt = 0.0
            if cnt <= 0:
                continue
            side = str(t.get("taker_side") or t.get("side") or "").lower()
            if side in {"yes", "buy"}:
                out.append(cnt)
            elif side in {"no", "sell"}:
                out.append(-cnt)
            else:
                out.append(cnt)
        return out

    def vpin_proxy(self, ticker: str, bucket: float = 400.0) -> float:
        seq = self.get_trade_signed_sizes(ticker, limit=300)
        if len(seq) < 5:
            return 0.0
        return float(vpin_from_signed_volumes(seq, bucket_target=bucket))

    def get_balance(self) -> float:
        """
        Available trading balance in dollars.

        Kalshi v2 returns ``balance`` in cents at the top level (not a nested object).
        """
        data = self._get("/portfolio/balance")
        if not isinstance(data, dict):
            return 0.0
        bd = data.get("balance_dollars")
        if bd is not None:
            try:
                return max(0.0, float(str(bd)))
            except (TypeError, ValueError):
                pass
        bal = data.get("balance")
        if isinstance(bal, (int, float)):
            return max(0.0, float(bal) / 100.0)
        if isinstance(bal, dict):
            for key in ("available_balance", "balance", "available"):
                v = bal.get(key)
                if isinstance(v, (int, float)):
                    return max(0.0, float(v) / 100.0)
        for key in ("available_balance", "balance"):
            v = data.get(key)
            if isinstance(v, (int, float)):
                return max(0.0, float(v) / 100.0)
        return 0.0


def attach_vpin_proxy_batch(
    client: object,
    market_lines: list[MarketLine],
    *,
    max_workers: int = 8,
) -> None:
    """Fill ``vpin_toxicity`` on each line using concurrent per-ticker fetches."""
    if not market_lines or not hasattr(client, "vpin_proxy"):
        return

    def _one(ml: MarketLine) -> tuple[str, float]:
        try:
            return ml.ticker, float(getattr(client, "vpin_proxy")(ml.ticker))
        except Exception:
            return ml.ticker, 0.0

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        futures = {ex.submit(_one, ml): ml for ml in market_lines}
        for fut in as_completed(futures):
            ml = futures[fut]
            try:
                _, tox = fut.result()
                ml.vpin_toxicity = float(tox)
            except Exception:
                ml.vpin_toxicity = 0.0


class MockKalshiClient:
    MOCK = [
        {"name": "Shohei Ohtani", "line": 1.5, "yes_ask": 0.55, "yes_bid": 0.53},
        {"name": "Aaron Judge", "line": 1.5, "yes_ask": 0.49, "yes_bid": 0.47},
        {"name": "Juan Soto", "line": 1.5, "yes_ask": 0.52, "yes_bid": 0.50},
    ]

    def get_total_bases_lines(self, game_date: Optional[str] = None, series_ticker: str = "KXMLBTB") -> list[MarketLine]:
        date_str = game_date or datetime.today().strftime("%Y-%m-%d")
        lines = []
        for i, p in enumerate(self.MOCK):
            ticker = f"MLB-TB-{p['name'].upper().replace(' ', '-')}-{date_str}"
            lines.append(
                MarketLine(
                    ticker=ticker,
                    player_name=p["name"],
                    player_id=i + 1,
                    game_date=date_str,
                    line=p["line"],
                    yes_ask=p["yes_ask"],
                    yes_bid=p["yes_bid"],
                    no_ask=round(1 - p["yes_bid"], 4),
                    no_bid=round(1 - p["yes_ask"], 4),
                    volume=500,
                    open_interest=1000,
                    xref_player_id="",
                    vpin_toxicity=0.0,
                )
            )
        if not game_date:
            return lines
        return [ml for ml in lines if ml.game_date == game_date]

    def place_order(self, ticker: str, side: str, contracts: int, price: float, order_type: str = "limit") -> OrderResult:
        log.info(f"[MOCK] {ticker} | {side.upper()} x{contracts} @ {price:.2f}")
        return OrderResult(True, f"mock-{int(datetime.now().timestamp())}", ticker, side, contracts, price, "mock")

    def get_orders(self, status: str = "resting", ticker: Optional[str] = None, limit: int = 200) -> list[OpenOrder]:
        return []

    def get_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "mock", "count": 0, "remaining_count": 0}

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_market(self, ticker: str) -> dict:
        return {"ticker": ticker, "result": ""}

    def get_balance(self) -> float:
        return 1000.0

    def get_trade_signed_sizes(self, ticker: str, limit: int = 200) -> list[float]:
        return []

    def vpin_proxy(self, ticker: str, bucket: float = 400.0) -> float:
        return 0.0


def get_client(force_mock: bool = False) -> KalshiClient | MockKalshiClient:
    if force_mock:
        return MockKalshiClient()
    key_id = str(KALSHI_API_KEY_ID).strip()
    key_path = str(KALSHI_PRIVATE_KEY_PATH).strip()
    if not key_id or not key_path:
        if REQUIRE_KALSHI_CREDENTIALS:
            raise RuntimeError(
                "Kalshi credentials missing: set KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH in .env "
                "(or unset REQUIRE_KALSHI_CREDENTIALS to allow the mock client)."
            )
        return MockKalshiClient()

    pem = Path(key_path).expanduser()
    if not pem.is_file():
        msg = f"Kalshi private key file not found: {pem}"
        if REQUIRE_KALSHI_CREDENTIALS:
            raise FileNotFoundError(msg)
        log.warning("%s — using mock Kalshi client.", msg)
        return MockKalshiClient()

    try:
        return KalshiClient()
    except OSError as e:
        if REQUIRE_KALSHI_CREDENTIALS:
            raise
        log.warning("Kalshi client init failed (%s); using mock.", e)
        return MockKalshiClient()
    except Exception as e:
        if REQUIRE_KALSHI_CREDENTIALS:
            raise
        log.warning("Kalshi client init failed (%s); using mock.", e, exc_info=True)
        return MockKalshiClient()

