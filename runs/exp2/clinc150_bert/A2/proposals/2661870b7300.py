import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Classifier-selected class-conditional neighbourhood distance detects queries that may be near the overall training manifold but are not locally supported by examples of the intent the classifier predicts. L2-normalized encoder representations emphasize semantic angular similarity rather than feature magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(ev.pooled.shape[0], dtype=float)
    fallback = ctx.h.knn(tr.pooled, ev.pooled, k=10, l2=True)

    for cls in np.unique(pred):
        ev_mask = pred == cls
        tr_mask = tr.labels == cls
        n_class = int(np.sum(tr_mask))
        if n_class >= 2:
            k = min(10, n_class)
            out[ev_mask] = ctx.h.knn(
                tr.pooled[tr_mask],
                ev.pooled[ev_mask],
                k=k,
                l2=True,
            )
        else:
            out[ev_mask] = fallback[ev_mask]

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
