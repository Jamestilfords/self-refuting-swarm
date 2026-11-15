import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The entropy of normalized Dirichlet concentrations measures the shape of evidential support "
    "independently of total evidence. Queries with an unusually diffuse or unusually peaked evidence "
    "shape for their predicted intent receive a high class-calibrated two-sided surprise score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    p_tr = alpha_tr / np.sum(alpha_tr, axis=1, keepdims=True)
    p_ev = alpha_ev / np.sum(alpha_ev, axis=1, keepdims=True)

    k = alpha_tr.shape[1]
    denom = max(np.log(float(k)), 1e-12)
    shape_tr = -np.sum(p_tr * np.log(np.maximum(p_tr, 1e-12)), axis=1) / denom
    shape_ev = -np.sum(p_ev * np.log(np.maximum(p_ev, 1e-12)), axis=1) / denom

    pred_ev = np.argmax(alpha_ev, axis=1)
    labels = np.asarray(tr.labels, dtype=int)

    global_sorted = np.sort(shape_tr)
    out = np.empty(shape_ev.shape[0], dtype=float)

    for c in range(k):
        idx = np.flatnonzero(pred_ev == c)
        if idx.size == 0:
            continue

        class_values = shape_tr[labels == c]
        if class_values.size >= 8:
            reference = np.sort(class_values)
        else:
            reference = global_sorted

        n = reference.size
        values = shape_ev[idx]
        lower = np.searchsorted(reference, values, side="right")
        upper = n - np.searchsorted(reference, values, side="left")
        tail = (np.minimum(lower, upper) + 1.0) / (n + 1.0)
        out[idx] = -np.log(np.maximum(tail, 1e-12))

    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=0.0)
