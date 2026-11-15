import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope predictions tend to have a clear winner over the nearest competing intent, whereas OOS inputs often have a smaller top-two logit margin. A temperature-scaled margin reduces sensitivity to global logit magnitude while preserving local classifier ambiguity."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=np.float64)
    eval_logits = np.asarray(ev.logits, dtype=np.float64)

    scale = np.median(
        np.max(train_logits, axis=1) - np.partition(train_logits, -2, axis=1)[:, -2]
    )
    scale = max(float(scale), 1e-6)

    top2 = np.partition(eval_logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    scores = -margin / scale

    if np.all(scores == scores[0]):
        scores = -np.max(eval_logits, axis=1) / max(
            float(np.std(np.max(train_logits, axis=1))), 1e-6
        )
    return scores
