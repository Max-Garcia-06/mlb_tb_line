from kalshi_bridge import MarketLine, attach_vpin_proxy_batch


class _Client:
    def vpin_proxy(self, ticker: str) -> float:
        return 0.42 if "A" in ticker else 0.1


def test_attach_vpin_proxy_batch():
    lines = [
        MarketLine("t-A", "P", 1, "2026-01-01", 1.5, 0.5, 0.48, 0.52, 0.5),
        MarketLine("t-B", "Q", 2, "2026-01-01", 2.5, 0.5, 0.48, 0.52, 0.5),
    ]
    attach_vpin_proxy_batch(_Client(), lines, max_workers=2)
    assert lines[0].vpin_toxicity == 0.42
    assert lines[1].vpin_toxicity == 0.1
