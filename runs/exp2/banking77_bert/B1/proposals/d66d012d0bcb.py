import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope representations should be distinctly closer to one intent prototype than to competing intents. "
    "Queries falling into geometrically ambiguous regions have an unusually small nearest-versus-runner-up class-distance gap."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0114"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_order = np.partition(train_d, 1, axis=1)
    eval_order = np.partition(eval_d, 1, axis=1)

    train_gap = (train_order[:, 1] - train_order[:, 0]) / (
        train_order[:, 1] + train_order[:, 0] + 1e-8
    )
    eval_gap = (eval_order[:, 1] - eval_order[:, 0]) / (
        eval_order[:, 1] + eval_order[:, 0] + 1e-8
    )

    nearest_train = np.argmin(train_d, axis=1)
    nearest_eval = np.argmin(eval_d, axis=1)
    scores = np.empty(eval_gap.shape[0], dtype=float)

    for c in range(train_d.shape[1]):
        reference = np.sort(train_gap[nearest_train == c])
        mask = nearest_eval == c
        if reference.size < 8:
            reference = np.sort(train_gap)
        ranks = np.searchsorted(reference, eval_gap[mask], side="right")
        scores[mask] = 1.0 - ranks / (reference.size + 1.0)

    return np.clip(scores, 0.0, 1.0)
