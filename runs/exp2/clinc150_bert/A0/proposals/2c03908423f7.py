import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should lie near one intent basin with a clear advantage over competing "
    "basins. OOS queries tend to be globally distant from all intent regions or geometrically "
    "ambiguous between multiple regions."
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

    train_near = train_two[:, 0]
    eval_near = eval_two[:, 0]

    train_gap = (train_two[:, 1] - train_two[:, 0]) / (
        train_two[:, 1] + train_two[:, 0] + 1e-8
    )
    eval_gap = (eval_two[:, 1] - eval_two[:, 0]) / (
        eval_two[:, 1] + eval_two[:, 0] + 1e-8
    )

    near_med = np.median(train_near)
    near_scale = np.percentile(train_near, 75) - np.percentile(train_near, 25)
    gap_med = np.median(train_gap)
    gap_scale = np.percentile(train_gap, 75) - np.percentile(train_gap, 25)

    near_scale = max(float(near_scale), 1e-6)
    gap_scale = max(float(gap_scale), 1e-6)

    distance_novelty = (eval_near - near_med) / near_scale
    basin_ambiguity = (gap_med - eval_gap) / gap_scale
    scores = 0.65 * distance_novelty + 0.35 * basin_ambiguity
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
