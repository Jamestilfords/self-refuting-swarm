import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can deviate along low-variance within-class directions that are suppressed by standard principal-subspace methods. A train-calibrated trailing-eigenspace score emphasizes these locally constrained directions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    detector = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_score = np.asarray(detector.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_score = np.asarray(detector.bottom(ev.pooled), dtype=float).reshape(-1)

    train_score = np.nan_to_num(train_score, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_score = np.nan_to_num(eval_score, nan=0.0, posinf=1e12, neginf=-1e12)

    center = np.median(train_score)
    scale = np.median(np.abs(train_score - center))
    if scale < 1e-10:
        scale = np.std(train_score)
    if scale < 1e-10:
        scale = 1.0

    return np.clip((eval_score - center) / scale, -1e6, 1e6)
