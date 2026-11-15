import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A query may be near the overall training manifold but unsupported by the intent selected by its classifier logits. Multi-radius class-conditional kNN distances in bottleneck space measure this local predicted-intent mismatch."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pred = np.argmax(ev.logits, axis=1)
    out = np.empty(len(ev.pooled), dtype=float)
    all_train = tr.feats

    for c in np.unique(pred):
        mask = pred == c
        class_train = tr.feats[tr.labels == c]
        if len(class_train) == 0:
            class_train = all_train

        k_small = min(3, len(class_train))
        k_large = min(15, len(class_train))
        d_small = ctx.h.knn(class_train, ev.feats[mask], k=k_small, l2=True)
        d_large = ctx.h.knn(class_train, ev.feats[mask], k=k_large, l2=True)
        out[mask] = 0.45 * d_small + 0.55 * d_large

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
