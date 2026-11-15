import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the per-item shared logit shift, in-scope examples tend to have a "
    "class-discriminative logit-shape norm supported by training data. Low centered-logit "
    "norm flags diffuse intent allocation independently of energy."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    eval_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)

    train_norm = np.sqrt(np.mean(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.mean(eval_centered * eval_centered, axis=1))

    order = np.sort(train_norm)
    ranks = np.searchsorted(order, eval_norm, side="right")
    support = ranks / float(max(order.size, 1))

    return 1.0 - support
