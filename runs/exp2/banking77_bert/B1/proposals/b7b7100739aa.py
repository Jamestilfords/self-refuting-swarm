import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Tempered energy emphasizes the broad logit shape rather than only the winning "
    "class confidence, making diffuse OOS evidence more distinguishable from in-scope peaks."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_span = np.max(tr.logits, axis=1) - np.min(tr.logits, axis=1)
    temperature = float(np.median(train_span))
    temperature = max(temperature, 1e-3)

    scaled = ev.logits / temperature
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    logsumexp = (
        max_scaled[:, 0]
        + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    )
    scores = -temperature * logsumexp

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
