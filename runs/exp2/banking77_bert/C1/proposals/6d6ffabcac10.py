import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Trailing within-class covariance directions emphasize residual deviations that dominant semantic directions suppress. Calibrating this trailing-subspace residual against the predicted class's training distribution makes unusual low-variance deviations more comparable across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_tail = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    train_tail = np.nan_to_num(
        train_tail, nan=0.0, posinf=1e12, neginf=-1e12
    )
    eval_tail = np.nan_to_num(
        eval_tail, nan=0.0, posinf=1e12, neginf=-1e12
    )

    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(np.asarray(ev.logits), axis=1)
    global_sorted = np.sort(train_tail)

    class_sorted = {}
    for c in np.unique(labels):
        vals = train_tail[labels == c]
        if vals.size >= 4:
            class_sorted[int(c)] = np.sort(vals)

    out = np.empty(eval_tail.shape[0], dtype=float)
    for i, (value, cls) in enumerate(zip(eval_tail, pred)):
        reference = class_sorted.get(int(cls), global_sorted)
        n = reference.size
        right = np.searchsorted(reference, value, side="right")
        survival = (n - right + 0.5) / (n + 1.0)
        out[i] = -np.log(max(survival, 1e-12))

    return np.nan_to_num(out, nan=0.0, posinf=30.0, neginf=0.0)
