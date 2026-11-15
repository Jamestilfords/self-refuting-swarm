import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Dirichlet mutual information measures disagreement induced by the concentration shape, "
    "separating low-evidence ambiguous predictions from confident in-scope posterior shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def dirichlet_mi(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        alpha0 = np.sum(alpha, axis=1, keepdims=True)
        mean = alpha / alpha0
        predictive_entropy = -np.sum(mean * np.log(np.maximum(mean, 1e-12)), axis=1)
        expected_entropy = -np.sum(
            mean * (digamma(alpha + 1.0) - digamma(alpha0 + 1.0)),
            axis=1,
        )
        return np.maximum(predictive_entropy - expected_entropy, 0.0)

    train_mi = dirichlet_mi(tr.logits)
    eval_mi = dirichlet_mi(ev.logits)

    train_pred = np.argmax(ctx.h.alpha(tr.logits), axis=1)
    eval_pred = np.argmax(ctx.h.alpha(ev.logits), axis=1)

    global_sorted = np.sort(train_mi)
    calibrated = np.empty(len(eval_mi), dtype=float)

    for cls in np.unique(eval_pred):
        mask = eval_pred == cls
        reference = np.sort(train_mi[train_pred == cls])
        if reference.size < 8:
            reference = global_sorted
        calibrated[mask] = (
            np.searchsorted(reference, eval_mi[mask], side="right") / reference.size
        )

    scale = max(float(np.mean(np.abs(train_mi))), 1e-6)
    return calibrated + 1e-3 * (eval_mi / scale)
