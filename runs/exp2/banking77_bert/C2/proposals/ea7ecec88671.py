import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often have a smaller separation between their top two class logits, reflecting ambiguity among in-scope intents. A normalized negative top-two margin reduces sensitivity to global logit scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    top2 = np.partition(logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    scale = np.sqrt(np.mean(logits * logits, axis=1) + 1e-12)
    return -margin / scale
