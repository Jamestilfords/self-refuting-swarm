import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often have a small gap between their two most competitive intent logits, reflecting ambiguity rather than a dominant in-scope intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return -margin
