"""
CLI entry for the MLB total-bases pipeline.

Implementation lives in ``pipeline/commands.py``; the Typer app is assembled in ``cli/main.py``.
"""

from cli.main import app

if __name__ == "__main__":
    app()
