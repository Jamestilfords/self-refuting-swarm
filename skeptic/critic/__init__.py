"""Public critic interface. Do not rename these exports.

Imports are lazy so ``python -m skeptic.critic.config`` does not load this
package's verdict code before the config module runs.
"""

__all__ = [
    "BaselineSet",
    "CriticConfig",
    "DEFAULT_CONFIG",
    "Decision",
    "ER1_COMMIT",
    "GateResult",
    "JsonDirStatsCache",
    "LordPlusPlus",
    "MemoryStatsCache",
    "ScoreSet",
    "StatsCache",
    "Verdict",
    "compute_baselines",
    "ece_in_scope",
    "load_scoreset",
    "verdict",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from skeptic.critic.baselines import compute_baselines, ece_in_scope
    from skeptic.critic.cache import JsonDirStatsCache, MemoryStatsCache, StatsCache
    from skeptic.critic.config import DEFAULT_CONFIG, CriticConfig
    from skeptic.critic.er1 import ER1_COMMIT
    from skeptic.critic.io import load_scoreset
    from skeptic.critic.lord import LordPlusPlus
    from skeptic.critic.types import BaselineSet, Decision, GateResult, ScoreSet, Verdict
    from skeptic.critic.verdict import verdict

    values = {
        "BaselineSet": BaselineSet,
        "CriticConfig": CriticConfig,
        "DEFAULT_CONFIG": DEFAULT_CONFIG,
        "Decision": Decision,
        "ER1_COMMIT": ER1_COMMIT,
        "GateResult": GateResult,
        "JsonDirStatsCache": JsonDirStatsCache,
        "LordPlusPlus": LordPlusPlus,
        "MemoryStatsCache": MemoryStatsCache,
        "ScoreSet": ScoreSet,
        "StatsCache": StatsCache,
        "Verdict": Verdict,
        "compute_baselines": compute_baselines,
        "ece_in_scope": ece_in_scope,
        "load_scoreset": load_scoreset,
        "verdict": verdict,
    }
    globals().update(values)
    return values[name]
