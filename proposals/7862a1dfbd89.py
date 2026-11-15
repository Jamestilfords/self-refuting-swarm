import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for supported intents. "
    "A classifier-selected, train-calibrated tail probability of trailing-space residual energy flags queries that violate this low-variance structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0241", "m_0188", "m_0220"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_bottom = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_bottom = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    sorted_scores = []
    for c in range(k):
        values = np.sort(train_bottom[train_pred == c])
        if values.size < 8:
            values = np.sort(train_bottom[tr.labels == c])
        if values.size == 0:
            values = np.sort(train_bottom)
        sorted_scores.append(values)

    out = np.empty(eval_bottom.shape[0], dtype=float)
    for c in range(k):
        mask = eval_pred == c
        if np.any(mask):
            reference = sorted_scores[c]
            ranks = np.searchsorted(reference, eval_bottom[mask], side="right")
            tail = (reference.size - ranks + 1.0) / (reference.size + 1.0)
            out[mask] = -np.log(np.maximum(tail, 1e-12))

    return out
