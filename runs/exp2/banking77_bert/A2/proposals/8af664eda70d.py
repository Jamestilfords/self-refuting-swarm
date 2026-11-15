import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Class-conditional k-nearest-neighbour distance in the normalized bottleneck space measures whether a query has local support among examples of the intent assigned by the classifier. OOS queries forced into a known intent should often remain distant from that intent's local representation manifold."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0049"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_x = ctx.h.l2(tr.feats)
    eval_x = ctx.h.l2(ev.feats)
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(eval_x), dtype=float)

    for cls in np.unique(pred):
        mask = pred == cls
        class_x = train_x[tr.labels == cls]
        if len(class_x) == 0:
            class_x = train_x
        k = min(10, len(class_x))
        out[mask] = ctx.h.knn(class_x, eval_x[mask], k=k, l2=False)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
