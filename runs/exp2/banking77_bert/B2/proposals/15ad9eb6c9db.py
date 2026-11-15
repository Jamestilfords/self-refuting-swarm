import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be close to some known intent while being implausible for the intent selected by "
    "the classifier. Calibrating predicted-class Mahalanobis distance against correctly classified "
    "training examples yields a class-specific tail score and exposes logit-feature disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_d.shape[1]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_pred = np.clip(train_pred, 0, n_classes - 1)
    eval_pred = np.clip(eval_pred, 0, n_classes - 1)
    labels = np.clip(np.asarray(tr.labels, dtype=int), 0, n_classes - 1)

    out = np.empty(eval_d.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if not np.any(mask):
            continue

        correct = (labels == c) & (train_pred == c)
        if np.sum(correct) < 8:
            correct = labels == c
        ref = np.sort(train_d[correct, c])

        if ref.size == 0:
            ref = np.sort(train_d[:, c])

        x = eval_d[mask, c]
        ranks = np.searchsorted(ref, x, side="right")
        tail = (ranks + 0.5) / (ref.size + 1.0)

        med = np.median(ref)
        mad = np.median(np.abs(ref - med)) + 1e-6
        local_excess = np.clip((x - med) / mad, -20.0, 20.0)

        out[mask] = np.log(tail / (1.0 - tail)) + 0.01 * local_excess

    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=-30.0)
