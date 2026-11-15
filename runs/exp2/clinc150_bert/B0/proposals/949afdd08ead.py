import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "A small top-two logit margin indicates that the classifier is ambiguous between competing intents, a logit-shape signal not captured by maximum probability alone. Temperature scaling is fitted on training logits through their typical margin scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_sorted = np.partition(tr.logits, -2, axis=1)[:, -2:]
    train_margin = train_sorted[:, 1] - train_sorted[:, 0]
    scale = float(np.median(train_margin))
    scale = max(scale, 1e-6)

    eval_sorted = np.partition(ev.logits, -2, axis=1)[:, -2:]
    margin = eval_sorted[:, 1] - eval_sorted[:, 0]
    score = -margin / scale

    norm_scale = float(np.median(np.linalg.norm(tr.logits, axis=1)))
    norm_scale = max(norm_scale, 1e-6)
    norm = np.linalg.norm(ev.logits, axis=1)
    score = score - 0.1 * norm / norm_scale
    return np.asarray(score, dtype=float)
