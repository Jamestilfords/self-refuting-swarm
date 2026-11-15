import numpy as np
from sklearn.neighbors import LocalOutlierFactor

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Local Outlier Factor compares each normalized pooled representation's neighborhood density with the densities of its neighbors. OOS queries should have sparse or inconsistent local support even when their nearest in-scope example is moderately close."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    n_neighbors = max(2, min(35, len(x_train) - 1))
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute",
        novelty=True,
        contamination="auto",
    )
    lof.fit(x_train)
    scores = -lof.score_samples(x_eval)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
