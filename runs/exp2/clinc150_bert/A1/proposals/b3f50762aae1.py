import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS representations should be both far from their closest known-intent prototype and weakly separated between the nearest competing prototypes. Dividing nearest-class distance by the nearest-versus-runner-up distance emphasizes this prototype ambiguity without using evaluation-set calibration."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    train_near = np.min(train_two, axis=1)
    train_second = np.max(train_two, axis=1)
    train_gap = np.maximum(train_second - train_near, 0.0)

    scale = float(np.median(train_gap[np.isfinite(train_gap)]))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    cap = float(np.percentile(train_near[np.isfinite(train_near)], 99.9))
    if not np.isfinite(cap) or cap <= 0.0:
        cap = 1.0

    eval_d = g.class_mahalanobis_sq(ev.pooled)
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    nearest = np.min(eval_two, axis=1)
    second = np.max(eval_two, axis=1)
    gap = np.maximum(second - nearest, 0.0)

    score = nearest / (gap + 0.1 * scale)
    return np.nan_to_num(score, nan=cap / (0.1 * scale), posinf=10.0 * cap / scale, neginf=0.0)
