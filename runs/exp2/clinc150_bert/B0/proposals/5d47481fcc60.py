import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope queries assigned to a given predicted intent tend to have a characteristic overall logit magnitude, while OOS queries often produce weak aggregate classifier evidence. Per-predicted-intent lower-tail calibration prevents intents with naturally smaller logit scales from being over-flagged."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_norm = np.sqrt(np.sum(tr.logits * tr.logits, axis=1))
    eval_norm = np.sqrt(np.sum(ev.logits * ev.logits, axis=1))

    global_sorted = np.sort(train_norm)
    scores = np.empty(eval_norm.shape[0], dtype=float)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        use = np.sort(train_norm[mask]) if np.any(mask) else global_sorted
        idx = eval_pred == c
        if np.any(idx):
            rank = np.searchsorted(use, eval_norm[idx], side="right")
            scores[idx] = 1.0 - (rank + 0.5) / (use.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
