import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope examples should have cosine alignment with the prototype of the classifier-predicted intent that is typical for that intent. A train-fitted, predicted-class-conditional lower-tail score detects semantic prototype mismatch rather than merely ranking by nearest prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    n_classes = tr.logits.shape[1]

    train_x = tr.feats / np.maximum(
        np.linalg.norm(tr.feats, axis=1, keepdims=True), eps
    )
    eval_x = ev.feats / np.maximum(
        np.linalg.norm(ev.feats, axis=1, keepdims=True), eps
    )

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, tr.labels, train_x)

    global_proto = np.mean(train_x, axis=0)
    global_proto /= max(float(np.linalg.norm(global_proto)), eps)

    valid = counts > 0
    prototypes[valid] /= counts[valid, None]
    prototypes[valid] /= np.maximum(
        np.linalg.norm(prototypes[valid], axis=1, keepdims=True), eps
    )
    prototypes[~valid] = global_proto

    train_own_cos = np.sum(train_x * prototypes[tr.labels], axis=1)
    sorted_own = [
        np.sort(train_own_cos[tr.labels == c]) if counts[c] > 0 else np.sort(train_own_cos)
        for c in range(n_classes)
    ]

    pred = np.argmax(ev.logits, axis=1)
    pred_cos = np.sum(eval_x * prototypes[pred], axis=1)
    scores = np.empty(pred_cos.shape[0], dtype=float)

    for c in range(n_classes):
        mask = pred == c
        if not np.any(mask):
            continue
        reference = sorted_own[c]
        lower_tail = np.searchsorted(reference, pred_cos[mask], side="right")
        lower_tail = lower_tail.astype(float) / float(reference.size)
        scores[mask] = 1.0 - lower_tail

    return np.clip(scores, 0.0, 1.0)
