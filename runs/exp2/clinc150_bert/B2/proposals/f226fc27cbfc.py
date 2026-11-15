import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Low-variance within-class directions encode constrained in-scope details, so unusual trailing-subspace "
    "energy is an OOS signal. Calibrating that energy within the classifier's predicted intent accounts for "
    "different residual scales across intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0117", "m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    model = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_tail = np.asarray(model.bottom(tr.pooled), dtype=float)
    eval_tail = np.asarray(model.bottom(ev.pooled), dtype=float)

    train_tail = np.nan_to_num(train_tail, nan=0.0, posinf=1e12, neginf=-1e12)
    eval_tail = np.nan_to_num(eval_tail, nan=0.0, posinf=1e12, neginf=-1e12)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    scores = np.empty(eval_tail.shape[0], dtype=float)

    global_sorted = np.sort(train_tail)
    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        reference = np.sort(train_tail[mask])
        if reference.size < 8:
            reference = global_sorted
        idx = np.searchsorted(reference, eval_tail[eval_pred == c], side="right")
        scores[eval_pred == c] = idx / float(reference.size + 1)

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
