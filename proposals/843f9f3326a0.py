import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A multiscale kNN expansion score detects queries that are not only far from known examples "
    "but also have unusually sparse local geometry between near and wider neighborhood radii."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    n = len(x_train)

    ref_mask = (np.arange(n) % 5) != 0
    if np.sum(ref_mask) < 2:
        ref_mask = np.ones(n, dtype=bool)
    cal_mask = ~ref_mask
    if np.sum(cal_mask) == 0:
        cal_mask = ref_mask.copy()

    reference = x_train[ref_mask]
    calibration = x_train[cal_mask]
    max_k = max(1, min(48, len(reference)))
    near_k = max(1, min(8, max_k))

    cal_near = ctx.h.knn(reference, calibration, k=near_k, l2=True)
    cal_wide = ctx.h.knn(reference, calibration, k=max_k, l2=True)
    ev_near = ctx.h.knn(reference, x_eval, k=near_k, l2=True)
    ev_wide = ctx.h.knn(reference, x_eval, k=max_k, l2=True)

    eps = 1e-12
    cal_radius = np.log(np.maximum(cal_wide, eps))
    ev_radius = np.log(np.maximum(ev_wide, eps))

    cal_expansion = np.log(
        np.maximum(cal_wide, eps) / np.maximum(cal_near, eps)
    )
    ev_expansion = np.log(
        np.maximum(ev_wide, eps) / np.maximum(ev_near, eps)
    )

    def upper_tail(values, reference_values):
        ordered = np.sort(np.asarray(reference_values, dtype=np.float64))
        ranks = np.searchsorted(ordered, values, side="right")
        p = (ranks + 1.0) / (len(ordered) + 2.0)
        return -np.log(np.maximum(1.0 - p, 1e-12))

    radius_surprise = upper_tail(ev_radius, cal_radius)
    expansion_surprise = upper_tail(ev_expansion, cal_expansion)
    scores = radius_surprise + 0.7 * expansion_surprise

    return np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=0.0)
