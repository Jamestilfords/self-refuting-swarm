import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Within-class trailing covariance directions are tightly constrained for known intents. "
    "A class-conditioned upper-tail calibration of trailing-direction residual energy highlights violations that are atypical for the classifier-selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0097", "m_0111"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    part = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_tail = np.asarray(part.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_tail = np.asarray(part.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    sorted_tail = []
    global_tail = np.sort(np.nan_to_num(train_tail, nan=0.0, posinf=1e12, neginf=-1e12))
    for c in range(n_classes):
        values = train_tail[train_pred == c]
        if values.size < 8:
            values = global_tail
        else:
            values = np.sort(np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12))
        sorted_tail.append(values)

    out = np.empty(eval_tail.shape[0], dtype=float)
    safe_eval = np.nan_to_num(eval_tail, nan=0.0, posinf=1e12, neginf=-1e12)
    for c in range(n_classes):
        mask = eval_pred == c
        if np.any(mask):
            ref = sorted_tail[c]
            out[mask] = np.searchsorted(ref, safe_eval[mask], side="right") / (ref.size + 1.0)

    return out
