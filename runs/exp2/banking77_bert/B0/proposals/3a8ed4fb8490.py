import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have a strongly dispersed logit vector after removing the arbitrary common logit offset. Low centered-logit norm indicates weak relative evidence across known intents and is therefore more likely to be out-of-scope."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    centered = logits - np.mean(logits, axis=1, keepdims=True)
    dispersion = np.sqrt(np.mean(centered * centered, axis=1))
    return -dispersion
