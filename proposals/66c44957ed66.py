import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The curvature of the centered log-partition function across temperatures captures how "
    "winner concentration changes as logits are softened. In-scope intents have characteristic "
    "class-conditional free-energy response shapes, while OOS queries often produce atypical curves."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0139"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def response_features(logits):
        z = np.asarray(logits, dtype=float)
        z = z - np.mean(z, axis=1, keepdims=True)
        temps = np.array([0.5, 1.0, 2.0, 4.0])
        parts = np.column_stack(
            [t * logsumexp(z / t, axis=1) for t in temps]
        )
        slope_1 = (parts[:, 1] - parts[:, 0]) / 0.5
        slope_2 = (parts[:, 2] - parts[:, 1]) / 1.0
        slope_3 = (parts[:, 3] - parts[:, 2]) / 2.0
        return np.column_stack([
            parts[:, 1],
            slope_1 - slope_2,
            slope_2 - slope_3,
        ])

    xtr = response_features(tr.logits)
    xev = response_features(ev.logits)
    k = tr.logits.shape[1]
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0) + 1e-8
    counts = np.bincount(pred_tr, minlength=k).astype(float)

    class_mean = np.empty((k, xtr.shape[1]), dtype=float)
    class_var = np.empty_like(class_mean)
    prior = 8.0

    for c in range(k):
        mask = pred_tr == c
        n = counts[c]
        if n > 0:
            xc = xtr[mask]
            mean_c = np.mean(xc, axis=0)
            var_c = np.var(xc, axis=0)
        else:
            mean_c = global_mean
            var_c = global_var

        weight = n / (n + prior)
        class_mean[c] = weight * mean_c + (1.0 - weight) * global_mean
        class_var[c] = weight * var_c + (1.0 - weight) * global_var + 1e-8

    delta = xev - class_mean[pred_ev]
    return np.sum((delta * delta) / class_var[pred_ev], axis=1)
