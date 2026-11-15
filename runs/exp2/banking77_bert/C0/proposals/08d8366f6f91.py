import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditional nearest-neighbor distance measures whether a query lies near examples of the class the classifier assigns to it, rather than merely near any in-scope example. OOS inputs that receive a confident but poorly supported class assignment should have larger within-predicted-class distances."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    predicted = np.argmax(ev.logits, axis=1)
    scores = np.empty(ev.logits.shape[0], dtype=float)
    k = 7

    for cls in np.unique(predicted):
        eval_mask = predicted == cls
        train_mask = tr.labels == cls
        class_train = tr.pooled[train_mask]

        if class_train.shape[0] >= k:
            scores[eval_mask] = ctx.h.knn(
                class_train,
                ev.pooled[eval_mask],
                k=k,
                l2=True,
            )
        else:
            scores[eval_mask] = ctx.h.knn(
                tr.pooled,
                ev.pooled[eval_mask],
                k=k,
                l2=True,
            )

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
