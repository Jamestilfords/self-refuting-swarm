import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query can be close to an in-scope class overall yet be atypical for the class selected by the classifier. Predicted-class Mahalanobis distance, calibrated against that class's training-distance tail, exposes this classifier-feature mismatch."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_d.shape[1]
    train_labels = np.clip(np.asarray(tr.labels, dtype=int), 0, n_classes - 1)
    pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    own_train_d = train_d[np.arange(train_d.shape[0]), train_labels]
    global_ref = np.sort(own_train_d)
    scores = np.empty(eval_d.shape[0], dtype=float)

    for c in range(n_classes):
        mask = pred == c
        if not np.any(mask):
            continue

        ref = train_d[train_labels == c, c]
        if ref.size < 2:
            ref = global_ref
        else:
            ref = np.sort(ref)

        ranks = np.searchsorted(ref, eval_d[mask, c], side="right")
        scores[mask] = (ranks + 0.5) / (ref.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
