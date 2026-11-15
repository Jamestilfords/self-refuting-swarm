import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent by the classifier should also be geometrically typical of that specific intent, not merely close to some other class prototype. Class-wise tail calibration accounts for intents with naturally different within-class distance scales."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0149"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_dist = g.class_mahalanobis_sq(tr.pooled)
    eval_dist = g.class_mahalanobis_sq(ev.pooled)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = train_dist.shape[1]
    pred = np.argmax(ev.logits, axis=1)

    references = []
    for c in range(n_classes):
        values = train_dist[labels == c, c]
        if values.size == 0:
            values = train_dist[:, c]
        references.append(np.sort(values))

    out = np.empty(eval_dist.shape[0], dtype=float)
    for c in range(n_classes):
        mask = pred == c
        if not np.any(mask):
            continue
        reference = references[c]
        values = eval_dist[mask, c]
        ranks = np.searchsorted(reference, values, side="right")
        out[mask] = ranks / float(reference.size + 1)

    return np.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
