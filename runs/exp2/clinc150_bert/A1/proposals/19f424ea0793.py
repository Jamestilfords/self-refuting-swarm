import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL out-of-scope inputs can distribute non-vacuous evidence more evenly across competing intents, "
    "creating high evidential dissonance even when total concentration is not unusual."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def dissonance(logits):
        alpha = np.asarray(ctx.h.alpha(logits), dtype=float)
        evidence = np.maximum(alpha - 1.0, 0.0)
        total = evidence.sum(axis=1, keepdims=True)
        uniform = np.full_like(evidence, 1.0 / evidence.shape[1])
        weights = np.divide(
            evidence,
            total,
            out=uniform,
            where=total > 1e-12,
        )
        return 1.0 - np.sum(weights * weights, axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_d = dissonance(tr.logits)
    eval_d = dissonance(ev.logits)

    scores = np.empty(len(eval_d), dtype=float)
    global_ref = np.sort(train_d)

    for c in range(tr.logits.shape[1]):
        ref = np.sort(train_d[train_pred == c])
        if ref.size < 2:
            ref = global_ref
        mask = eval_pred == c
        ranks = np.searchsorted(ref, eval_d[mask], side="right")
        scores[mask] = ranks / (ref.size + 1.0)

    return np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
