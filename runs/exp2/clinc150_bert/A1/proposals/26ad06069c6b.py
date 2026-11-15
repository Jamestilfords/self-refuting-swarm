import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope examples should be substantially closer to one class mean than to competing class means. OOS queries often have a smaller nearest-versus-runner-up Mahalanobis gap, indicating ambiguous compatibility with known intents."
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

    train_gap = train_two[:, 1] - train_two[:, 0]
    eval_gap = eval_two[:, 1] - eval_two[:, 0]

    reference = np.sort(train_gap)
    cdf = np.searchsorted(reference, eval_gap, side="right") / float(reference.size)
    return 1.0 - cdf
