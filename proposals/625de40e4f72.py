import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The Dirichlet mutual-information gap between predictive entropy and expected categorical entropy "
    "depends on the detailed concentration shape, not just energy or vacuity. A train-fitted, "
    "predicted-intent-specific relationship with total evidence and predictive entropy flags atypical "
    "evidential geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def evidence_features(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        alpha = np.clip(alpha, 1e-8, 1e12)
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        h_mean = -np.sum(p * np.log(np.clip(p, 1e-12, 1.0)), axis=1)

        expected_entropy = (
            digamma(total + 1.0)
            - np.sum(p * digamma(alpha + 1.0), axis=1)
        )
        mutual_info = np.maximum(h_mean - expected_entropy, 0.0)

        x = np.column_stack(
            [
                np.ones_like(total),
                np.log1p(total),
                h_mean,
                h_mean * h_mean,
            ]
        )
        return x, mutual_info

    x_tr, y_tr = evidence_features(tr.logits)
    x_ev, y_ev = evidence_features(ev.logits)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]
    d = x_tr.shape[1]
    ridge = 1e-3

    def fit_group(x, y):
        gram = x.T @ x
        penalty = ridge * np.eye(d)
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(gram + penalty, x.T @ y)
        resid = y - x @ beta
        scale = np.sqrt(np.mean(resid * resid) + 1e-10)
        return beta, scale

    global_beta, global_scale = fit_group(x_tr, y_tr)
    betas = np.tile(global_beta, (k, 1))
    scales = np.full(k, global_scale)

    for c in range(k):
        mask = pred_tr == c
        if np.sum(mask) >= 12:
            beta, scale = fit_group(x_tr[mask], y_tr[mask])
            betas[c] = beta
            scales[c] = scale

    pred_mi = np.sum(x_ev * betas[pred_ev], axis=1)
    residual = np.abs(y_ev - pred_mi) / np.maximum(scales[pred_ev], 1e-6)

    return np.asarray(residual, dtype=float)
