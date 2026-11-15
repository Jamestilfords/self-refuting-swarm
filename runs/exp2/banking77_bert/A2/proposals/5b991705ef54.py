import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The centered logit norm measures how strongly the classifier concentrates evidence across intents, "
    "independently of the common logit offset used by energy. OOS inputs often have unusually weak class-specific "
    "logit concentration even when their maximum probability is not extreme."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def centered_norm(logits):
        z = np.asarray(logits, dtype=float)
        z = z - np.mean(z, axis=1, keepdims=True)
        return np.sqrt(np.mean(z * z, axis=1))

    train_norm = centered_norm(tr.logits)
    eval_norm = centered_norm(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_sorted = np.sort(train_norm)
    scores = np.empty(eval_norm.shape[0], dtype=float)

    for c in np.unique(eval_pred):
        reference = train_norm[train_pred == c]
        if reference.size < 8:
            reference = train_norm
        reference = np.sort(reference)
        mask = eval_pred == c
        ranks = np.searchsorted(reference, eval_norm[mask], side="right")
        scores[mask] = (reference.size - ranks + 0.5) / (reference.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=0.0, neginf=1.0)
