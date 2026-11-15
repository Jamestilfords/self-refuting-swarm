import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Dirichlet mutual information measures disagreement induced by the full concentration shape, "
    "so balanced competing evidence is flagged even when total evidence alone is not unusually low. "
    "Predicted-intent empirical calibration accounts for classes with intrinsically different evidential shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def dirichlet_mi(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=np.float64)
        alpha = np.clip(alpha, 1e-8, 1e12)
        total = np.sum(alpha, axis=1, keepdims=True)
        prob = alpha / total

        predictive_entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-12)), axis=1)
        expected_entropy = (
            digamma(total[:, 0] + 1.0)
            - np.sum(prob * digamma(alpha + 1.0), axis=1)
        )
        return np.maximum(predictive_entropy - expected_entropy, 0.0)

    train_mi = dirichlet_mi(tr.logits)
    eval_mi = dirichlet_mi(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_ref = np.sort(train_mi)
    scores = np.empty(len(eval_mi), dtype=np.float64)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        ref = np.sort(train_mi[mask]) if np.any(mask) else global_ref
        idx = eval_pred == c
        if np.any(idx):
            scores[idx] = np.searchsorted(ref, eval_mi[idx], side="right") / float(len(ref))

    return scores
