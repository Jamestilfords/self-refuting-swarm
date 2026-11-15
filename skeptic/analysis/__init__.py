"""Truth, false-discovery metrics, and Experiment 1 tables.

Owner: @mid9t (Ryu). ``exp2.py`` is Experiment 2. ``scorecard.py`` is a separate
package. The public metric names live in ``skeptic.analysis.truth``.
"""

__all__ = [
    "AnalysisConfig",
    "DEFAULT",
    "fdp",
    "paired_bootstrap_fast",
    "recall",
    "true_improvement",
    "winners_curse",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in {"AnalysisConfig", "DEFAULT"}:
        from skeptic.analysis.config import DEFAULT, AnalysisConfig

        values = {"AnalysisConfig": AnalysisConfig, "DEFAULT": DEFAULT}
    elif name == "paired_bootstrap_fast":
        from skeptic.analysis.bootstrap import paired_bootstrap_fast

        values = {"paired_bootstrap_fast": paired_bootstrap_fast}
    else:
        from skeptic.analysis.truth import fdp, recall, true_improvement, winners_curse

        values = {
            "fdp": fdp,
            "recall": recall,
            "true_improvement": true_improvement,
            "winners_curse": winners_curse,
        }
    globals().update(values)
    return values[name]
