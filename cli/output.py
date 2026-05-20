"""Shared Rich console helpers for CLI commands."""

from rich.console import Console
from rich.panel import Panel

console = Console()


def _header(title: str):
    console.print(Panel(f"[bold cyan]{title}[/bold cyan]", expand=False))


def _warn(msg: str):
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def _success(msg: str):
    console.print(f"[bold green]✓[/bold green] {msg}")
