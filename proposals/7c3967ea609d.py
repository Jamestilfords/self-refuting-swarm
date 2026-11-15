import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should also be geometrically competitive with the nearest "
    "feature-space intent prototype. The ratio of predicted-class Mahalanobis distance to the "
    "nearest-class distance flags classifier-geometry disagreement, calibrated separately for each predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0112"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = d_tr.shape[1]

    pred_tr = np.clip(pred_tr, 0, n_classes - 1)
    pred_ev = np.clip(pred_ev, 0, n_classes - 1)

    row_tr = np.arange(d_tr.shape[0])
    row_ev = np.arange(d_ev.shape[0])

    selected_tr = d_tr[row_tr, pred_tr]
    selected_ev = d_ev[row_ev, pred_ev]
    nearest_tr = np.min(d_tr, axis=1)
    nearest_ev = np.min(d_ev, axis=1)

    ratio_tr = selected_tr / (nearest_tr + 1e-12)
    ratio_ev = selected_ev / (nearest_ev + 1e-12)
    ratio_tr = np.nan_to_num(ratio_tr, nan=1.0, posinf=1e12, neginf=1.0)
    ratio_ev = np.nan_to_num(ratio_ev, nan=1.0, posinf=1e12, neginf=1.0)

    global_ref = np.sort(ratio_tr)
    out = np.empty(ratio_ev.shape[0], dtype=float)

    for c in range(n_classes):
        mask_tr = pred_tr == c
        mask_ev = pred_ev == c
        ref = np.sort(ratio_tr[mask_tr]) if np.any(mask_tr) else global_ref
        left = np.searchsorted(ref, ratio_ev[mask_ev], side="left")
        right = np.searchsorted(ref, ratio_ev[mask_ev], side="right")
        p = (left + right) / (2.0 * (ref.size + 1.0))
        out[mask_ev] = -np.log1p(-p)

    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=0.0)
