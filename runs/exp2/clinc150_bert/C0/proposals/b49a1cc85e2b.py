import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can be close to some class while still being atypical for the class selected by the classifier. A class-conditional Mahalanobis tail probability calibrated on training examples penalizes such prediction-distance disagreement."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_d_all = g.class_mahalanobis_sq(ev.pooled)

    n_classes = train_d_all.shape[1]
    labels = np.clip(np.asarray(tr.labels, dtype=int), 0, n_classes - 1)
    pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_d = train_d_all[np.arange(train_d_all.shape[0]), labels]
    eval_d = eval_d_all[np.arange(eval_d_all.shape[0]), pred]

    sorted_by_class = []
    for c in range(n_classes):
        values = np.sort(train_d[labels == c])
        if values.size == 0:
            values = np.sort(train_d)
        sorted_by_class.append(values)

    scores = np.empty(eval_d.shape[0], dtype=float)
    for c in range(n_classes):
        mask = pred == c
        if not np.any(mask):
            continue
        reference = sorted_by_class[c]
        rank = np.searchsorted(reference, eval_d[mask], side="right")
        tail = (reference.size - rank + 1.0) / (reference.size + 1.0)
        scores[mask] = -np.log(tail) + 1e-8 * (
            eval_d[mask] / (1.0 + np.maximum(eval_d[mask], 0.0))
        )

    return np.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=0.0)
