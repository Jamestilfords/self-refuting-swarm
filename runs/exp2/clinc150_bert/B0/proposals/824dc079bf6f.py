import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions capture tightly constrained in-scope structure that can be violated by near-OOS queries. "
    "Calibrating trailing-subspace residuals by predicted intent accounts for different intrinsic residual scales across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(model.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_tail = np.asarray(model.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(len(eval_tail), dtype=float)

    global_sorted = np.sort(train_tail)
    global_scale = np.median(np.abs(train_tail - np.median(train_tail))) + 1e-8

    for c in np.unique(eval_pred):
        reference = train_tail[train_pred == c]
        if reference.size < 8:
            reference = train_tail[tr.labels == c]
        if reference.size < 8:
            reference = train_tail

        reference = np.sort(reference)
        ranks = np.searchsorted(reference, eval_tail[eval_pred == c], side="right")
        percentile = ranks / (reference.size + 1.0)

        center = np.median(reference)
        residual = (eval_tail[eval_pred == c] - center) / global_scale
        scores[eval_pred == c] = percentile + 1e-6 * residual

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
