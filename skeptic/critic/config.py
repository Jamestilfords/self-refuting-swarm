"""Every critic threshold. Gates read this object and do not hard-code numbers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from scipy.stats import norm

# Proposed defaults (†). Ryu confirms or replaces them in PREREGISTRATION.md.
DAGGER = frozenset({
    "g2_rho_thresh",
    "g2_weak_incumbent_check",
    "bootstrap_B",
    "fdr_w0",
    "mdd_power",
    "g6_tol",
})


@dataclass(frozen=True)
class CriticConfig:
    g1_flip_thresh: float = 0.001
    g2_rho_thresh: float = 0.999  # †
    g2_weak_incumbent_check: bool = True  # †
    bootstrap_B: int = 2000  # †
    bootstrap_seed: int = 0
    fdr_alpha: float = 0.1
    fdr_w0: float = 0.05  # †
    g4_band: float = 0.001
    mdd_power: float = 0.80  # mdd_z = z_0.95 + z_power ≈ 2.4865 †
    g5_ece_tol: float = 0.01
    g5_bins: int = 15
    g6_angles: int = 720
    g6_tol: float = 0.0001  # †
    seed_majority: str = "ceil_half"  # 3 of 5

    @property
    def mdd_z(self) -> float:
        return float(norm.ppf(0.95) + norm.ppf(self.mdd_power))

    def digest(self) -> str:
        payload = {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}
        raw = json.dumps(payload, sort_keys=True, allow_nan=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def markdown(self) -> str:
        lines = ["| key | value |", "| --- | --- |"]
        for field in dataclasses.fields(self):
            mark = " †" if field.name in DAGGER else ""
            lines.append(f"| {field.name} | {getattr(self, field.name)}{mark} |")
        lines.append(f"| mdd_z | {self.mdd_z:.6f} † |")
        return "\n".join(lines)


DEFAULT_CONFIG = CriticConfig()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Print CriticConfig for the pre-registration.")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    if args.markdown:
        print(DEFAULT_CONFIG.markdown())
    else:
        parser.error("pass --markdown")


if __name__ == "__main__":
    main()
