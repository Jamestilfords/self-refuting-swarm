"""LORD++ online FDR (Ramdas, Yang, Wainwright, Jordan 2017).

gamma is the onlineFDR default, not renormalized to sum to 1:

    gamma(j) = 0.07720838 * log(max(j, 2)) / (j * exp(sqrt(log j)))

for j >= 1, with natural log. Wealth update, with missing terms equal to 0:

    alpha_t = gamma(t) * w0
              + (alpha - w0) * gamma(t - tau_1)
              + alpha * sum_{j >= 2} gamma(t - tau_j)
"""

from __future__ import annotations

import math


def gamma(j: int) -> float:
    if j < 1:
        return 0.0
    index = float(j)
    return 0.07720838 * math.log(max(index, 2.0)) / (index * math.exp(math.sqrt(math.log(index))))


class LordPlusPlus:
    def __init__(self, alpha: float = 0.1, w0: float = 0.05):
        if not (0.0 < w0 <= alpha < 1.0):
            raise ValueError("require 0 < w0 <= alpha < 1")
        self.alpha = float(alpha)
        self.w0 = float(w0)
        self.t = 0
        self.rejections: tuple[int, ...] = ()

    def level(self) -> float:
        """alpha_{t+1}. Does not mutate."""
        step = self.t + 1
        wealth = gamma(step) * self.w0
        if not self.rejections:
            return wealth
        first = self.rejections[0]
        wealth += (self.alpha - self.w0) * gamma(step - first)
        wealth += self.alpha * sum(gamma(step - tau) for tau in self.rejections[1:])
        return wealth

    def test(self, p: float) -> bool:
        """Reject when p <= alpha_t. Always consumes one step."""
        alpha_t = self.level()
        self.t += 1
        reject = float(p) <= alpha_t
        if reject:
            self.rejections = self.rejections + (self.t,)
        return reject

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "w0": self.w0,
            "t": self.t,
            "rejections": list(self.rejections),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LordPlusPlus":
        state = cls(alpha=payload["alpha"], w0=payload["w0"])
        state.t = int(payload["t"])
        state.rejections = tuple(int(tau) for tau in payload["rejections"])
        return state

    def copy(self) -> "LordPlusPlus":
        return LordPlusPlus.from_dict(self.to_dict())
