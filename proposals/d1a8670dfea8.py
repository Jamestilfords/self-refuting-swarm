import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query should be locally supported by examples from the intent selected by the classifier, "
    "not merely by the in-scope corpus as a whole. Averaging class-conditional normalized kNN "
    "radii at local and broader scales reduces sensitivity to accidental close neighbors."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0016", "m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n = len(tr.labels)
    idx = np.arange(n)

    cal_mask = (idx % 5) == 0
    ref_mask = ~cal_mask
    if not np.any(ref_mask):
        ref_mask = (idx % 2) == 0
        cal_mask = ~ref_mask
    if not np.any(cal_mask):
        cal_mask = ref_mask.copy()

    x_ref = tr.pooled[ref_mask]
    y_ref = tr.labels[ref_mask]
    x_cal = tr.pooled[cal_mask]
    y_cal = tr.labels[cal_mask]

    k_classes = tr.logits.shape[1]
    pred_eval = np.argmax(ev.logits, axis=1)
    out = np.empty(len(ev.logits), dtype=float)

    global_ref = x_ref
    global_cal = x_cal
    if len(global_ref) == 0:
        global_ref = tr.pooled
    if len(global_cal) == 0:
        global_cal = tr.pooled

    for c in range(k_classes):
        ref_c = x_ref[y_ref == c]
        cal_c = x_cal[y_cal == c]

        if len(ref_c) < 2:
            ref_c = global_ref
        if len(cal_c) == 0:
            cal_c = global_cal

        m = len(ref_c)
        k_near = max(1, min(8, m))
        k_broad = max(1, min(32, m))

        cal_near = ctx.h.knn(ref_c, cal_c, k=k_near, l2=True)
        cal_broad = ctx.h.knn(ref_c, cal_c, k=k_broad, l2=True)
        scale_near = max(float(np.median(cal_near)), 1e-8)
        scale_broad = max(float(np.median(cal_broad)), 1e-8)

        take = pred_eval == c
        if np.any(take):
            q = ev.pooled[take]
            d_near = ctx.h.knn(ref_c, q, k=k_near, l2=True)
            d_broad = ctx.h.knn(ref_c, q, k=k_broad, l2=True)
            out[take] = 0.5 * (d_near / scale_near + d_broad / scale_broad)

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)
