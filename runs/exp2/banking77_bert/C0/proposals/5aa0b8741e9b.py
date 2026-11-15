import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often induce less decisive overall logit geometry, producing a smaller centered logit norm than confidently recognized in-scope intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    centered = logits - np.mean(logits, axis=1, keepdims=True)
    norm = np.sqrt(np.sum(centered * centered, axis=1))
    return -norm
