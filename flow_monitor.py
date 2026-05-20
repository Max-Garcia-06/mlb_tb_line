"""
VPIN-style flow toxicity from coarse order-flow imbalance.

Kalshi does not always expose tick-level prints; this implementation accepts a sequence
of (signed_volume, bucket_target) and returns a toxicity score in [0, 1].
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable


@dataclass
class VpinState:
    bucket_volume: float = 0.0
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    vpin: float = 0.0
    buckets: Deque[float] = field(default_factory=lambda: deque(maxlen=50))

    def update(self, signed_size: float, bucket_target: float) -> float:
        """
        signed_size > 0 treated as aggressive buy flow, < 0 as sell.
        When cumulative |size| in the current bucket exceeds bucket_target, finalize VPIN increment.
        """
        tgt = max(float(bucket_target), 1.0)
        s = float(signed_size)
        self.bucket_volume += abs(s)
        if s >= 0:
            self.buy_vol += s
        else:
            self.sell_vol += -s
        while self.bucket_volume >= tgt:
            tot = self.buy_vol + self.sell_vol
            imb = abs(self.buy_vol - self.sell_vol)
            bvpin = imb / tot if tot > 0 else 0.0
            self.buckets.append(bvpin)
            self.vpin = sum(self.buckets) / len(self.buckets) if self.buckets else 0.0
            self.bucket_volume -= tgt
            self.buy_vol *= 0.5
            self.sell_vol *= 0.5
        return float(self.vpin)


def vpin_from_signed_volumes(volumes: Iterable[float], bucket_target: float = 500.0) -> float:
    st = VpinState()
    for v in volumes:
        st.update(float(v), bucket_target)
    return float(st.vpin)
