## MLB Total Bases Edge Pipeline

This project mirrors the `nba_reb_line` pipeline, but for **MLB player total bases (TB)**.

### What it does
- **ETL**: Pull batter game logs from MLB Stats API into SQLite.
- **Train**: Fit an XGBoost regression model to predict expected total bases (\(\lambda\)).
- **Scan**: Pull Kalshi TB markets for a given date, compute \(P(\text{TB} > k)\), detect +EV edges, and (optionally) place orders.
- **Report**: Journal placed orders and summarize performance by probability bucket once markets settle.

### Quickstart
1. Install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

2. Build dataset and model

```bash
python3 run_pipeline.py etl
python3 run_pipeline.py train
python3 run_pipeline.py evaluate
```

3. Dry-run scan (no orders)

```bash
python3 run_pipeline.py scan --dry-run
```

4. Live scan (places orders)

```bash
python3 run_pipeline.py scan --live --max-signals 5 --one-per-player
```

### Notes
- This code assumes Kalshi provides a TB series with titles like `"Player Name: 2+ total bases"`.
- Total bases is an integer count; most markets are half-point equivalent (e.g. `2+` → line \(k=1.5\)).

