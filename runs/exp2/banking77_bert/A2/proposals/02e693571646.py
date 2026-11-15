import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should have a clearly preferred Gaussian class, whereas OOS "
    "queries often have similar distances to multiple classes. A small nearest-versus-second-nearest "
    "class distance gap therefore signals geometric class ambiguity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]

    train_near = np.min(train_two, axis=1)
    train_second = np.max(train_two, axis=1)
    eval_near = np.min(eval_two, axis=1)
    eval_second = np.max(eval_two, axis=1)

    eps = 1e-8
    train_gap = np.log1p(np.maximum(train_second - train_near, 0.0) / (train_near + eps))
    eval_gap = np.log1p(np.maximum(eval_second - eval_near, 0.0) / (eval_near + eps))

    gap_med = float(np.median(train_gap))
    gap_q1, gap_q3 = np.percentile(train_gap, [25.0, 75.0])
    gap_scale = max(float(gap_q3 - gap_q1), 1e-6)

    near_med = float(np.median(train_near))
    near_q1, near_q3 = np.percentile(train_near, [25.0, 75.0])
    near_scale = max(float(near_q3 - near_q1), 1e-6)

    ambiguity = 1.0 / (1.0 + np.exp(np.clip((eval_gap - gap_med) / gap_scale, -50.0, 50.0)))
    remoteness = 1.0 / (
        1.0 + np.exp(np.clip(-(eval_near - near_med) / near_scale, -50.0, 50.0))
    )
    scores = ambiguity + 0.05 * remoteness
    return np.nan_to_num(scores, nan=0.0, posinf=1.05, neginf=0.0)
