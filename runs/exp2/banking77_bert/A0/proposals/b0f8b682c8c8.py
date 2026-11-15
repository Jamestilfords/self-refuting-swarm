import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query's logit-predicted intent should also be its closest class prototype in "
    "whitened representation space. A large predicted-class distance relative to the best competing "
    "prototype signals classifier-representation disagreement characteristic of OOS inputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    distances = g.class_mahalanobis_sq(ev.pooled)

    pred = np.argmax(ev.logits, axis=1)
    rows = np.arange(pred.shape[0])
    predicted_distance = distances[rows, pred]

    competitors = distances.copy()
    competitors[rows, pred] = np.inf
    best_competitor = np.min(competitors, axis=1)

    return predicted_distance - best_competitor
