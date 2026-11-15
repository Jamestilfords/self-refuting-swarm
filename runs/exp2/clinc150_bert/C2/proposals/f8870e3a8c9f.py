import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries can have a small top-two separation relative to the overall spread of their class logits, indicating ambiguous class preference rather than merely low absolute confidence. Normalizing the top-two margin by per-item logit dispersion makes this distinct from raw margin and energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    centered = logits - np.mean(logits, axis=1, keepdims=True)
    spread = np.sqrt(np.mean(centered * centered, axis=1))
    return -margin / (spread + 1e-8)
