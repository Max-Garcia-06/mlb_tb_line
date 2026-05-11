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

# Kalshi — RSA key auth
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "b1348fe1-4f3d-433b-ab9f-424d5839a777")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "/Users/maxgarcia/.kalshi/mlbtwin.pem")
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_ORDER_URL = os.getenv("KALSHI_ORDER_URL", "https://api.elections.kalshi.com/trade-api/v2")

# Edge detection
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.05"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.10"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.02"))

# Tail-risk controls
MIN_P = float(os.getenv("MIN_P", "0.20"))
TAIL_P_CUTOFF = float(os.getenv("TAIL_P_CUTOFF", "0.30"))
TAIL_EDGE_MULT = float(os.getenv("TAIL_EDGE_MULT", "2.0"))

# Execution guardrails
MIN_LIMIT_PRICE = float(os.getenv("MIN_LIMIT_PRICE", "0.20"))  # avoid tiny-probability longshots by default
MAX_YES_LINE = float(os.getenv("MAX_YES_LINE", "2.0"))  # e.g. avoid YES on 2.5+ if desired (set high to disable)

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

