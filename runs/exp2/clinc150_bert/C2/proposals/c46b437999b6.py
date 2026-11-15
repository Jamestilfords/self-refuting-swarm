import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope queries tend to produce a more pronounced class-discriminative logit vector, while OOS queries often have lower centered logit magnitude even when their common logit offset is large."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr = ctx.train
    ev = ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_centered = train_logits - np.mean(train_logits, axis=1, keepdims=True)
    scale = np.sqrt(np.mean(train_centered * train_centered, axis=1))
    reference = np.median(scale)
    reference = max(float(reference), 1e-8)

    centered = eval_logits - np.mean(eval_logits, axis=1, keepdims=True)
    magnitude = np.sqrt(np.mean(centered * centered, axis=1))
    return -np.log1p(magnitude / reference)
