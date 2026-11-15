import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An OOS query may be near some known class but atypical for the class selected by the classifier. "
    "Predicted-class Mahalanobis distance, calibrated against that class's training distances, measures this representation-logit disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0066", "m_0077"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    labels = np.asarray(tr.labels, dtype=int)
    n_classes = train_d.shape[1]
    pred = np.clip(np.argmax(ev.logits, axis=1).astype(int), 0, n_classes - 1)

    own_train_d = train_d[np.arange(labels.shape[0]), labels]
    score_out = np.empty(eval_d.shape[0], dtype=float)

    for c in range(n_classes):
        mask = pred == c
        if not np.any(mask):
            continue
        reference = np.sort(own_train_d[labels == c])
        if reference.size == 0:
            reference = np.sort(own_train_d)
        ranks = np.searchsorted(reference, eval_d[mask, c], side="right")
        score_out[mask] = ranks / float(reference.size + 1)

    return np.nan_to_num(score_out, nan=0.5, posinf=1.0, neginf=0.0)
