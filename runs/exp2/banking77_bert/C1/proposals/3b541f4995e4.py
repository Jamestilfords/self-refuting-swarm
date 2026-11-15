import numpy as np
from scipy.special import logsumexp

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Tempered energy emphasizes the breadth of moderately plausible competing intents rather than only the largest logit. OOS queries often retain a diffuse set of competing logits, which can separate them from confident in-scope predictions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    centered = train_logits - np.mean(train_logits, axis=1, keepdims=True)
    spread = np.sqrt(np.mean(centered * centered, axis=1))
    temperature = float(np.median(spread))
    temperature = max(temperature, 1e-3)

    scores = -temperature * logsumexp(eval_logits / temperature, axis=1)
    return np.asarray(scores, dtype=float)
