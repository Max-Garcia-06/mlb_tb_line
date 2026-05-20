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
SEASONS = [2022, 2023, 2024, 2025]

# Rolling window (games) for trailing features
ROLLING_WINDOW = 15

# Minimum games for a player to include in training
MIN_GAMES = 20

# Walk-forward CV controls (split by unique game_date)
CV_GAP_DATES = int(os.getenv("CV_GAP_DATES", "1"))  # embargo gap between train and test folds (in unique dates)

# Evaluation lines (TB) for probability scoring
EVAL_LINES = [float(x) for x in os.getenv("EVAL_LINES", "0.5,1.5,2.5,3.5").split(",")]

# Distribution to use: "poisson" or "nbinom"
DISTRIBUTION = os.getenv("DISTRIBUTION", "nbinom")

