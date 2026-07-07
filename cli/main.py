"""Typer application entry: registers all pipeline commands."""

from __future__ import annotations

import logging
import uuid

import typer

from pipeline import commands as cmd
from structured_logging import configure_structured_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
configure_structured_logging(run_id=str(uuid.uuid4())[:8])

app = typer.Typer(add_completion=False, help="MLB Total Bases Edge Pipeline")

app.command()(cmd.etl)
app.command("etl-statcast")(cmd.etl_statcast)
app.command()(cmd.train)
app.command()(cmd.evaluate)
app.command()(cmd.tune)
app.command()(cmd.scan)
app.command()(cmd.mark)
app.command()(cmd.report)
app.command("report-range")(cmd.report_range)
app.command()(cmd.reconcile)
app.command()(cmd.calibrate)
app.command("fit-blend")(cmd.fit_blend)
app.command("segment-report")(cmd.segment_report)
app.command()(cmd.snapshot)
app.command("schedule-snapshots")(cmd.schedule_snapshots)
app.command()(cmd.backtest)
app.command("materialize-features")(cmd.materialize_features)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
