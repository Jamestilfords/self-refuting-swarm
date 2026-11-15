import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope query should lie near the Gaussian mean of the class favored by its classifier logits. OOS queries and representation-logit disagreements receive high scores when their predicted-class Mahalanobis distance is unusually large relative to that class's training distribution."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    classes = np.arange(train_dist.shape[1])
    own_train = train_dist[np.arange(len(tr.labels)), tr.labels]

    sums = np.bincount(tr.labels, weights=own_train, minlength=len(classes))
    counts = np.bincount(tr.labels, minlength=len(classes)).astype(float)
    means = sums / np.maximum(counts, 1.0)

    centered_sq = (own_train - means[tr.labels]) ** 2
    vars_ = (
        np.bincount(tr.labels, weights=centered_sq, minlength=len(classes))
        / np.maximum(counts - 1.0, 1.0)
    )
    scales = np.sqrt(np.maximum(vars_, 1e-6))

    pred = np.argmax(ev.logits, axis=1)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)
    pred_dist = eval_dist[np.arange(len(pred)), pred]

    score_values = (pred_dist - means[pred]) / scales[pred]
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)
