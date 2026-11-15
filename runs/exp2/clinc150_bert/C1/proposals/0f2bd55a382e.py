import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low maximum softmax probability captures classifier uncertainty, while nearest-neighbor distance in normalized encoder space captures local feature-space novelty. Each component is mapped to its training empirical percentile before combination, so neither raw scale dominates."
SOURCE = ""
PARENTS = ["msp"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_uncertainty = 1.0 - np.max(ctx.h.softmax(tr.logits), axis=1)
    eval_uncertainty = 1.0 - np.max(ctx.h.softmax(ev.logits), axis=1)

    train_features = ctx.h.l2(tr.pooled)
    eval_features = ctx.h.l2(ev.pooled)
    train_distance = ctx.h.knn(train_features, train_features, k=10, l2=False)
    eval_distance = ctx.h.knn(train_features, eval_features, k=10, l2=False)

    def percentile_from_train(train_values, values):
        train_values = np.nan_to_num(
            np.asarray(train_values, dtype=float),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        values = np.nan_to_num(
            np.asarray(values, dtype=float),
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )
        reference = np.sort(train_values)
        ranks = np.searchsorted(reference, values, side="right")
        return (ranks + 0.5) / (reference.size + 1.0)

    uncertainty_pct = percentile_from_train(train_uncertainty, eval_uncertainty)
    distance_pct = percentile_from_train(train_distance, eval_distance)

    return 0.35 * uncertainty_pct + 0.65 * distance_pct
