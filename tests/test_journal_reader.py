from pathlib import Path

from journal_reader import date_from_journal_filename, load_jsonl_rows, placed_post_submit


def test_date_from_journal_filename():
    assert date_from_journal_filename(Path("trades_2026-05-01.jsonl")) == "2026-05-01"
    assert date_from_journal_filename(Path("other.jsonl")) is None


def test_load_jsonl_rows_missing(tmp_path):
    p = tmp_path / "missing.jsonl"
    assert load_jsonl_rows(p) == []


def test_placed_post_submit_filters():
    rows = [
        {"note": "post-submit", "success": True},
        {"note": "post-submit", "success": False},
        {"note": "fill"},
    ]
    assert len(placed_post_submit(rows)) == 1
