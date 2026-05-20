"""Typer application entry: registers all pipeline commands."""

from __future__ import annotations

import logging

import typer

from pipeline import commands as cmd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = typer.Typer(add_completion=False, help="MLB Total Bases Edge Pipeline")

app.command()(cmd.etl)
app.command()(cmd.train)
app.command()(cmd.evaluate)
app.command()(cmd.tune)
app.command()(cmd.scan)
app.command()(cmd.mark)
app.command()(cmd.report)
app.command("report-range")(cmd.report_range)
app.command()(cmd.reconcile)
app.command()(cmd.calibrate)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
