from segment_health import SegmentMetrics, build_segment_health_report, judge_segment


def test_judge_segment_pass():
    m = SegmentMetrics(
        side="yes",
        line="1.5",
        spread="<0.05",
        edge="0.10-0.14",
        orders=10,
        fills=6,
        contracts=30,
        cost=12.0,
        realized_pnl=2.0,
        clv_sum=1.5,
        clv_contracts=30,
    )
    v = judge_segment(m)
    assert v.status == "PASS"


def test_judge_segment_fail_low_clv(monkeypatch):
    monkeypatch.setattr("segment_health.SEGMENT_MIN_FILLS", 3)
    monkeypatch.setattr("segment_health.SEGMENT_MIN_FILL_RATE", 0.2)
    monkeypatch.setattr("segment_health.SEGMENT_MIN_AVG_CLV", 0.0)
    monkeypatch.setattr("segment_health.SEGMENT_MIN_ROI_PCT", -100.0)
    m = SegmentMetrics(
        side="no",
        line="1.5",
        spread="0.05-0.09",
        edge=">=0.20",
        orders=8,
        fills=5,
        contracts=20,
        cost=10.0,
        realized_pnl=0.0,
        clv_sum=-2.0,
        clv_contracts=20,
    )
    v = judge_segment(m)
    assert v.status == "FAIL"
    assert any("avg_clv" in r for r in v.reasons)


def test_build_report_trade_vs_pause():
    passing = SegmentMetrics(
        side="yes",
        line="1.5",
        spread="<0.05",
        edge="0.10-0.14",
        orders=10,
        fills=6,
        contracts=30,
        cost=12.0,
        clv_sum=1.0,
        clv_contracts=30,
    )
    report = build_segment_health_report(
        {passing.segment_key: passing},
    )
    assert report.recommendation == "TRADE"

    weak = SegmentMetrics(
        side="no",
        line="2.5",
        spread=">=0.20",
        edge="<0.05",
        orders=2,
        fills=1,
    )
    report2 = build_segment_health_report({weak.segment_key: weak})
    assert report2.recommendation == "PAUSE"
