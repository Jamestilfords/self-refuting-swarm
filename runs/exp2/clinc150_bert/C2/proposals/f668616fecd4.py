import numpy as np
from scipy.special import digamma, gammaln

FAMILY = "other"
ARM = "edl"
RATIONALE = "Dirichlet differential entropy is high for diffuse, balanced concentration vectors and low for sharp class-specific evidence. Class-conditional robust calibration makes unusual evidential shapes score highly without being a monotone transform of logit energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    def dirichlet_entropy(a):
        total = np.sum(a, axis=1)
        k = a.shape[1]
        value = (
            np.sum(gammaln(a), axis=1)
            - gammaln(total)
            + (total - k) * digamma(total)
            - np.sum((a - 1.0) * digamma(a), axis=1)
        )
        return np.nan_to_num(value, nan=0.0, posinf=1e12, neginf=-1e12)

    h_tr = dirichlet_entropy(a_tr)
    h_ev = dirichlet_entropy(a_ev)
    labels = tr.labels.astype(int)
    pred_ev = np.argmax(a_ev, axis=1)

    global_med = np.median(h_tr)
    global_scale = np.median(np.abs(h_tr - global_med)) * 1.4826
    if not np.isfinite(global_scale) or global_scale < 1e-6:
        global_scale = np.std(h_tr) + 1e-6

    n_classes = a_tr.shape[1]
    centers = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        values = h_tr[labels == c]
        if values.size:
            center = np.median(values)
            scale = np.median(np.abs(values - center)) * 1.4826
            if values.size < 3 or not np.isfinite(scale) or scale < 1e-6:
                scale = global_scale
            centers[c] = center
            scales[c] = scale

    return (h_ev - centers[pred_ev]) / (scales[pred_ev] + 1e-6)
