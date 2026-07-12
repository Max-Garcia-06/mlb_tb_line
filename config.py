import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
# Only load env vars from this project folder (avoid picking up ../.env)
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Kalshi — RSA key auth (empty defaults; live client requires both set in .env)
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
# If true, missing Kalshi credentials raise instead of falling back to MockKalshiClient.
REQUIRE_KALSHI_CREDENTIALS = os.getenv("REQUIRE_KALSHI_CREDENTIALS", "").lower() in ("1", "true", "yes")
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_ORDER_URL = os.getenv("KALSHI_ORDER_URL", "https://api.elections.kalshi.com/trade-api/v2")

# Predictive core: default ordinal logit; set USE_LEGACY_XGB=1 for XGB Tweedie + Poisson/NB head
USE_LEGACY_XGB = os.getenv("USE_LEGACY_XGB", "").lower() in ("1", "true", "yes")

# Market blend: shrink model probabilities toward the market mid in logit space.
# w=1 trusts the model fully (pre-July-2026 behavior); w=0 trusts the market.
# Effective weight: BLEND_WEIGHT env > fitted models/blend_meta.json > DEFAULT_BLEND_WEIGHT.
USE_MARKET_BLEND = os.getenv("USE_MARKET_BLEND", "true").lower() in ("1", "true", "yes")
_BLEND_W_RAW = os.getenv("BLEND_WEIGHT")
BLEND_WEIGHT_OVERRIDE = float(_BLEND_W_RAW) if _BLEND_W_RAW not in (None, "") else None
DEFAULT_BLEND_WEIGHT = float(os.getenv("DEFAULT_BLEND_WEIGHT", "0.35"))
# Floor on the fitted weight so a fit never fully zeroes the model out. Kept small:
# full-slate scoring (model-vs-market, 04-27->07-06, N=5794) confirmed the low fitted
# w is real signal, not small-sample noise (market beats model in every disagreement
# bucket and most days), so the floor should not manufacture trust the data doesn't
# support. A 0.3 floor did exactly that on 2026-07-10 and re-armed trading on edges
# the model has never earned.
MIN_BLEND_WEIGHT = float(os.getenv("MIN_BLEND_WEIGHT", "0.05"))
MIN_BLEND_ROWS = int(os.getenv("MIN_BLEND_ROWS", "200"))
BLEND_META_PATH = MODEL_DIR / "blend_meta.json"

# Segment-level blend weights (by |model-market| disagreement bucket; see
# market_blend.disagreement_bucket). Fit via `fit-blend-segments` on full-slate
# snapshot scoring (not fills - fills are edge-selected, so low-disagreement
# buckets would have near-zero fill coverage). Falls back to the global weight
# above when a segment is missing or under MIN_BLEND_ROWS_SEGMENT.
MIN_BLEND_ROWS_SEGMENT = int(os.getenv("MIN_BLEND_ROWS_SEGMENT", "100"))
SEGMENT_BLEND_META_PATH = MODEL_DIR / "blend_meta_segments.json"

# `refit-blend` trailing window: re-fits global + segment weights from full-slate
# scoring on a recurring schedule (see scripts/cron_job.sh) so w tracks the model's
# actual recent performance vs. the market instead of staying pinned wherever it was
# last set by hand. LAG_DAYS skips the most recent days, whose boxscores/ETL may not
# be finalized yet.
BLEND_REFIT_LOOKBACK_DAYS = int(os.getenv("BLEND_REFIT_LOOKBACK_DAYS", "30"))
BLEND_REFIT_LAG_DAYS = int(os.getenv("BLEND_REFIT_LAG_DAYS", "2"))

# Kalshi trading fees: taker fee = ceil(rate * C * P * (1-P)) cents; resting (maker) orders free.
KALSHI_TAKER_FEE_RATE = float(os.getenv("KALSHI_TAKER_FEE_RATE", "0.07"))
KALSHI_MAKER_FEE_RATE = float(os.getenv("KALSHI_MAKER_FEE_RATE", "0.0"))

# Maker mode: rest limit orders one tick inside the ask instead of crossing (no taker fee).
MAKER_MODE = os.getenv("MAKER_MODE", "true").lower() in ("1", "true", "yes")

# Segments blocked from live signals, as "line:side" pairs (line-1.5 NO lost -11.7% ROI
# on 417 resolved orders through 2026-07-05). Empty string disables.
BLOCKED_SEGMENTS = frozenset(
    tuple(seg.strip().lower().split(":", 1))
    for seg in os.getenv("BLOCKED_SEGMENTS", "1.5:no").split(",")
    if ":" in seg
)

# Edge detection
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.05"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.10"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.02"))
# Total deployed across all signals in one scan (defaults to MAX_BET_PCT for backward compatibility).
# Set lower than MAX_BET_PCT to force proportional shrink across legs when Kelly sums above this cap.
_MAX_PORT_PCT_RAW = os.getenv("MAX_PORTFOLIO_PCT")
MAX_PORTFOLIO_PCT = (
    float(_MAX_PORT_PCT_RAW) if _MAX_PORT_PCT_RAW not in (None, "") else MAX_BET_PCT
)

# Simultaneous Kelly / correlated slate (cvxpy QP; off by default — proportional cap-only sizing)
USE_CVX_PORTFOLIO = os.getenv("USE_CVX_PORTFOLIO", "").lower() in ("1", "true", "yes")
PORTFOLIO_RISK_AVERSION = float(os.getenv("PORTFOLIO_RISK_AVERSION", "2.5"))
SAME_SLATE_CORR = float(os.getenv("SAME_SLATE_CORR", "0.35"))
CROSS_SLATE_CORR = float(os.getenv("CROSS_SLATE_CORR", "0.04"))

# VPIN-style flow guard (0 = off; typical toxic threshold 0.65–0.80)
ENABLE_VPIN_GUARD = os.getenv("ENABLE_VPIN_GUARD", "true").lower() in ("1", "true", "yes")
VPIN_MAX_TOXIC = float(os.getenv("VPIN_MAX_TOXIC", "0.72"))

# Tail-risk controls
MIN_P = float(os.getenv("MIN_P", "0.20"))
TAIL_P_CUTOFF = float(os.getenv("TAIL_P_CUTOFF", "0.30"))
TAIL_EDGE_MULT = float(os.getenv("TAIL_EDGE_MULT", "2.0"))

# Kelly haircut for p_model bands with a history of poor calibration (win rate
# trailing model probability). Multiplier applies only within [LOW, HIGH).
RISKY_BAND_LOW = float(os.getenv("RISKY_BAND_LOW", "0.6"))
RISKY_BAND_HIGH = float(os.getenv("RISKY_BAND_HIGH", "0.9"))
RISKY_BAND_KELLY_MULT = float(os.getenv("RISKY_BAND_KELLY_MULT", "0.4"))

# Execution guardrails
MIN_LIMIT_PRICE = float(os.getenv("MIN_LIMIT_PRICE", "0.20"))  # avoid tiny-probability longshots by default
MAX_YES_LINE = float(os.getenv("MAX_YES_LINE", "2.0"))  # e.g. avoid YES on 2.5+ if desired (set high to disable)

# Isotonic fill calibrator: coherent P(under)=1-P(over); bounded so it cannot collapse to 0
USE_ISOTONIC_CALIBRATION = os.getenv("USE_ISOTONIC_CALIBRATION", "true").lower() in ("1", "true", "yes")
MAX_CALIB_P_DELTA = float(os.getenv("MAX_CALIB_P_DELTA", "0.12"))  # max |Δp| from raw per side

# Ignore "edges" against one-sided / penny asks (no real liquidity)
MIN_REALISTIC_ASK = float(os.getenv("MIN_REALISTIC_ASK", "0.06"))
MAX_REALISTIC_ASK = float(os.getenv("MAX_REALISTIC_ASK", "0.97"))

# Live scan: if model E[TB] exceeds this and player has thin history, fall back to market-implied λ
LAMBDA_SANITY_MAX = float(os.getenv("LAMBDA_SANITY_MAX", "4.25"))
GAMES_FOR_LAMBDA_SANITY = int(os.getenv("GAMES_FOR_LAMBDA_SANITY", "50"))

# Storage
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "mlb_tb.db"))

# Seasons (years) to pull
SEASONS = [int(x) for x in os.getenv("SEASONS", "2022,2023,2024,2025,2026").split(",") if x.strip()]

# Statcast (Baseball Savant via pybaseball): quality-of-contact tables + chunked pull cache
STATCAST_BATTER_TABLE = os.getenv("STATCAST_BATTER_TABLE", "statcast_batter_games")
STATCAST_PITCHER_TABLE = os.getenv("STATCAST_PITCHER_TABLE", "statcast_pitcher_games")
STATCAST_CACHE_DIR = Path(os.getenv("STATCAST_CACHE_DIR", str(DATA_DIR / "statcast_cache")))
STATCAST_CHUNK_DAYS = int(os.getenv("STATCAST_CHUNK_DAYS", "14"))
# Minimum batted-ball events for a batter-game statcast row to be kept (drops 0-BBE noise rows)
STATCAST_MIN_BBE = int(os.getenv("STATCAST_MIN_BBE", "1"))

# MLB Stats API retries (ETL / schedule)
MLB_API_MAX_RETRIES = int(os.getenv("MLB_API_MAX_RETRIES", "5"))
MLB_API_RETRY_SLEEP_SEC = float(os.getenv("MLB_API_RETRY_SLEEP_SEC", "2.0"))

# Rolling window (games) for trailing features
ROLLING_WINDOW = 15

# Minimum games for a player to include in training
MIN_GAMES = 20

# Include Statcast quality-of-contact rolling features in MODEL_FEATURES.
# Default off: on the current linear/GBM heads they did not improve walk-forward
# Brier/log-loss (collinear with existing per-AB rate features). The ETL and
# feature joins still run; flip this to experiment (e.g. with a richer model).
USE_STATCAST_FEATURES = os.getenv("USE_STATCAST_FEATURES", "").lower() in ("1", "true", "yes")

# Walk-forward CV controls (split by unique game_date)
CV_GAP_DATES = int(os.getenv("CV_GAP_DATES", "1"))  # embargo gap between train and test folds (in unique dates)

# Evaluation lines (TB) for probability scoring and CV model selection
EVAL_LINES = [float(x) for x in os.getenv("EVAL_LINES", "0.5,1.5,2.5,3.5").split(",")]

# Hard gate: require positive expected value (per contract) to emit a signal
MIN_EV = float(os.getenv("MIN_EV", "0.0"))

# Segmented isotonic calibration (line × side × games_played bucket)
MIN_CALIB_ROWS_GLOBAL = int(os.getenv("MIN_CALIB_ROWS_GLOBAL", "50"))
MIN_CALIB_ROWS_SEGMENT = int(os.getenv("MIN_CALIB_ROWS_SEGMENT", "30"))
USE_SEGMENTED_CALIBRATION = os.getenv("USE_SEGMENTED_CALIBRATION", "true").lower() in ("1", "true", "yes")

# Walk-forward model selection: mean_brier (default) or mean_logloss
CV_PRIMARY_METRIC = os.getenv("CV_PRIMARY_METRIC", "mean_brier").strip().lower()

# Distribution to use: "poisson" or "nbinom"
DISTRIBUTION = os.getenv("DISTRIBUTION", "nbinom")

# Live scan + backtest: only include games starting within this many hours (0 = full pre-game
# slate). Tightened 2 → 1.5 on 2026-07-06: fills marked 120m later showed -0.107/contract CLV
# vs +0.021 at 30m — early entries were getting run over by later lineup/pitcher info.
SCAN_WITHIN_HOURS = float(os.getenv("SCAN_WITHIN_HOURS", "1.5"))

# Segment health (segment-report go/no-go)
SEGMENT_MIN_FILLS = int(os.getenv("SEGMENT_MIN_FILLS", "5"))
SEGMENT_MIN_FILL_RATE = float(os.getenv("SEGMENT_MIN_FILL_RATE", "0.25"))
SEGMENT_MIN_AVG_CLV = float(os.getenv("SEGMENT_MIN_AVG_CLV", "0.0"))
SEGMENT_MIN_ROI_PCT = float(os.getenv("SEGMENT_MIN_ROI_PCT", "-5.0"))
SEGMENT_REPORT_LOOKBACK_DAYS = int(os.getenv("SEGMENT_REPORT_LOOKBACK_DAYS", "14"))

# Fill calibrator preflight (OOF still takes priority at inference when enabled)
CALIBRATE_MAX_AGE_DAYS = int(os.getenv("CALIBRATE_MAX_AGE_DAYS", "14"))
REQUIRE_FILL_CALIB_FOR_LIVE = os.getenv("REQUIRE_FILL_CALIB_FOR_LIVE", "").lower() in ("1", "true", "yes")

# Live risk desk
USE_LIVE_BALANCE = os.getenv("USE_LIVE_BALANCE", "true").lower() in ("1", "true", "yes")
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "0"))  # 0 = disabled
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0"))  # 0 = disabled; overrides _USD when set
MAX_ORDERS_PER_DAY = int(os.getenv("MAX_ORDERS_PER_DAY", "0"))  # 0 = disabled
MAX_DAILY_DEPLOYED_USD = float(os.getenv("MAX_DAILY_DEPLOYED_USD", "0"))  # 0 = disabled
MAX_DAILY_DEPLOYED_PCT = float(os.getenv("MAX_DAILY_DEPLOYED_PCT", "0"))  # 0 = disabled; overrides _USD when set
MAX_OPEN_RESTING_USD = float(os.getenv("MAX_OPEN_RESTING_USD", "0"))  # 0 = disabled
MAX_OPEN_RESTING_PCT = float(os.getenv("MAX_OPEN_RESTING_PCT", "0"))  # 0 = disabled; overrides _USD when set
MAX_CONTRACTS_PER_GAME = int(os.getenv("MAX_CONTRACTS_PER_GAME", "0"))  # 0 = disabled
MAX_LEGS_PER_PLAYER_DAY = int(os.getenv("MAX_LEGS_PER_PLAYER_DAY", "0"))  # 0 = disabled
RESERVE_RESTING_FROM_BANKROLL = os.getenv("RESERVE_RESTING_FROM_BANKROLL", "true").lower() in (
    "1",
    "true",
    "yes",
)
AUTO_KILL_ON_RISK_BREACH = os.getenv("AUTO_KILL_ON_RISK_BREACH", "true").lower() in ("1", "true", "yes")
KILL_SWITCH_PATH = Path(os.getenv("KILL_SWITCH_PATH", str(DATA_DIR / "KILL_SWITCH")))

# OOF isotonic calibration (fit from walk-forward CV; separate from fill-based calibrator)
USE_OOF_CALIBRATION = os.getenv("USE_OOF_CALIBRATION", "true").lower() in ("1", "true", "yes")

# Backtest
BACKTEST_PIT_TRAIN = os.getenv("BACKTEST_PIT_TRAIN", "true").lower() in ("1", "true", "yes")
BACKTEST_PIT_RETRAIN_DAYS = int(os.getenv("BACKTEST_PIT_RETRAIN_DAYS", "7"))
BACKTEST_USE_EARLIEST_SNAPSHOT = os.getenv("BACKTEST_USE_EARLIEST_SNAPSHOT", "true").lower() in ("1", "true", "yes")
BACKTEST_FILL_MODEL = os.getenv("BACKTEST_FILL_MODEL", "true").lower() in ("1", "true", "yes")

# Feature materialization (gold layer in SQLite)
MATERIALIZE_FEATURES_ON_ETL = os.getenv("MATERIALIZE_FEATURES_ON_ETL", "true").lower() in ("1", "true", "yes")
FEATURES_TABLE = os.getenv("FEATURES_TABLE", "batter_features")

# Scan summary email (Gmail SMTP; app password, not the account password)
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
GMAIL_SENDER = os.getenv("GMAIL_SENDER", "")
SCAN_EMAIL_RECIPIENT = os.getenv("SCAN_EMAIL_RECIPIENT", GMAIL_SENDER)
SCAN_EMAIL_ENABLED = os.getenv("SCAN_EMAIL_ENABLED", "true").lower() in ("1", "true", "yes")

