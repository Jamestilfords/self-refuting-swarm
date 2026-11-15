import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be close to training examples from the intent selected by the classifier, "
    "whereas OOS or confidently misrouted queries tend to have poor same-predicted-intent neighborhood support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(ev.feats), dtype=float)

    for c in np.unique(pred):
        mask = pred == c
        members = tr.feats[tr.labels == c]

        if len(members) == 0:
            members = tr.feats

        k = min(20, len(members))
        scores[mask] = ctx.h.knn(members, ev.feats[mask], k=k, l2=True)

    return np.nan_to_num(scores, nan=0.0, posinf=2.0, neginf=0.0)
