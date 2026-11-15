import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope queries often have a smaller gap between their two most supported intents, even when their overall logit scale resembles in-scope examples. A train-calibrated negative top-two margin identifies such ambiguous predictions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_top2 = np.partition(train_logits, -2, axis=1)[:, -2:]
    train_margin = train_top2[:, 1] - train_top2[:, 0]
    center = np.median(train_margin)
    scale = np.median(np.abs(train_margin - center)) + 1e-8

    eval_top2 = np.partition(eval_logits, -2, axis=1)[:, -2:]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]
    scores = (center - eval_margin) / scale
    return np.clip(scores, -1e6, 1e6)
