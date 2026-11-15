import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class eigen-directions capture tightly constrained variations of known intents. "
    "Calibrating their deviation within the classifier-predicted intent makes the score sensitive to "
    "class-specific low-variance violations rather than reproducing a single global trailing-distance ranking."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0115"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_raw = np.asarray(model.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_raw = np.asarray(model.bottom(ev.pooled), dtype=float).reshape(-1)

    train_raw = np.nan_to_num(train_raw, nan=0.0, posinf=1e6, neginf=-1e6)
    eval_raw = np.nan_to_num(eval_raw, nan=0.0, posinf=1e6, neginf=-1e6)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_raw)
    scores = np.empty(eval_raw.shape[0], dtype=float)

    for c in np.unique(eval_pred):
        mask_eval = eval_pred == c
        reference = train_raw[train_pred == c]

        if reference.size < 8:
            reference = global_sorted
        else:
            reference = np.sort(reference)

        ranks = np.searchsorted(reference, eval_raw[mask_eval], side="right")
        scores[mask_eval] = ranks / (reference.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
