import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Cosine agreement with the prototype of the classifier-selected intent detects representations whose predicted label is not supported by the corresponding angular feature geometry. Class-conditional training calibration makes low agreement unusual relative to examples receiving the same predicted intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_similarity = np.sum(x_train * prototypes[train_pred], axis=1)
    eval_similarity = np.sum(x_eval * prototypes[eval_pred], axis=1)

    sorted_by_pred = []
    fallback = np.sort(train_similarity)
    for c in range(n_classes):
        values = np.sort(train_similarity[train_pred == c])
        sorted_by_pred.append(values if values.size else fallback)

    tail = np.empty(eval_similarity.shape[0], dtype=float)
    for c in range(n_classes):
        mask = eval_pred == c
        if np.any(mask):
            reference = sorted_by_pred[c]
            rank = np.searchsorted(reference, eval_similarity[mask], side="right")
            tail[mask] = 1.0 - rank / float(reference.size)

    return tail + 0.05 * (1.0 - eval_similarity)
