import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The gap between the largest and second-largest logits measures decision-boundary ambiguity that is not identical to energy or maximum softmax probability. OOS queries often lack a dominant intent logit and therefore have small top-two margins."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_sorted = np.partition(train_logits, -2, axis=1)
    train_margin = train_sorted[:, -1] - train_sorted[:, -2]
    scale = np.median(train_margin)
    scale = max(float(scale), 1e-6)

    eval_sorted = np.partition(eval_logits, -2, axis=1)
    eval_margin = eval_sorted[:, -1] - eval_sorted[:, -2]
    scores = -eval_margin / scale
    return np.asarray(scores, dtype=float)
