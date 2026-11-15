import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope queries tend to have a larger gap between their most likely and runner-up classes. A small top-two logit margin indicates ambiguous class evidence and is scored as more out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return -margin
