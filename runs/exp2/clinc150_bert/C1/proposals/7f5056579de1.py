import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have a larger gap between their most likely and runner-up classes, while ambiguous OOS queries often have a small top-two logit margin. Normalizing the margin by logit scale reduces sensitivity to per-example confidence magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    centered = logits - np.mean(logits, axis=1, keepdims=True)
    scale = np.sqrt(np.mean(centered * centered, axis=1) + 1e-8)
    return -margin / scale
