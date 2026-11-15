import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The centered logit norm measures decisiveness of the full relative logit configuration while removing the common logit offset that drives energy. OOS queries may have unusually weak class-separating evidence even when their absolute logit scale is unremarkable."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0035"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    eval_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)

    train_norm = np.sqrt(np.sum(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.sum(eval_centered * eval_centered, axis=1))

    center = np.median(train_norm)
    scale = np.median(np.abs(train_norm - center))
    scale = max(float(scale), 1e-6)

    return (center - eval_norm) / scale
