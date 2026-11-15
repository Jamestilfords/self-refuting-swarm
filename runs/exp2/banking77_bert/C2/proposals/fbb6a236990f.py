import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often have a smaller gap between their two most supported intents, even when their overall logit energy is high. The negative top-two logit margin therefore flags ambiguous intent assignments."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    logits = np.asarray(ctx.eval.logits, dtype=float)
    if logits.shape[1] < 2:
        return -np.abs(logits[:, 0]) + 1e-12 * np.arange(logits.shape[0])
    top2 = np.partition(logits, logits.shape[1] - 2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return -margin
