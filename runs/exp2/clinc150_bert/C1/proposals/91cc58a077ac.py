import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This measures whether the Dirichlet concentration composition is atypical for the class favored by the evidential model, after removing total concentration. It therefore captures evidence-shape anomalies that are not a monotone transformation of energy or vacuity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    # Centered log-ratios retain the shape of the Dirichlet while discarding
    # its overall strength, making this distinct from vacuity-like scores.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    k = tr.logits.shape[1]
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    sums = np.zeros((k, z_tr.shape[1]), dtype=float)
    np.add.at(sums, tr.labels, z_tr)
    means = sums / np.maximum(counts[:, None], 1.0)

    residuals = z_tr - means[tr.labels]
    var = np.mean(residuals * residuals, axis=0)
    var = np.maximum(var, 1e-5)

    pred = np.argmax(a_ev, axis=1)
    diff = z_ev - means[pred]
    scores = np.mean((diff * diff) / var, axis=1)

    # A tiny deterministic shape-derived term breaks accidental ties without
    # altering the principal class-conditional anomaly ordering.
    scores = scores + 1e-10 * np.sum(z_ev * z_ev, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
