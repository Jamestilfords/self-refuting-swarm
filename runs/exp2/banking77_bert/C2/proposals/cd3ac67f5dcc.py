import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to produce a more dispersed class-logit profile, while OOS inputs often have relatively uniform logits after removing the arbitrary common logit offset. Negative centered logit norm measures this shape independently of energy's sensitivity to the common offset."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    centered = logits - np.mean(logits, axis=1, keepdims=True)
    return -np.sqrt(np.sum(centered * centered, axis=1) + 1e-12)
