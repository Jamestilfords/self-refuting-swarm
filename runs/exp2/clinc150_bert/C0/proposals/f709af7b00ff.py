import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have a larger gap between their most likely and second-most likely intents. A small top-two logit margin flags ambiguous predictions that are more likely to be out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    if logits.shape[1] < 2:
        return -np.abs(logits[:, 0])
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return -margin
