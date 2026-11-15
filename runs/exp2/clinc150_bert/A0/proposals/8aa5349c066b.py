import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two centered-logit margin measures whether one intent is decisively preferred over its closest competitor, independently of shared logit shifts captured by energy. OOS queries often have unusually small margins despite potentially moderate overall logit scale."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0095"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_centered = train_logits - train_logits.mean(axis=1, keepdims=True)
    eval_centered = eval_logits - eval_logits.mean(axis=1, keepdims=True)

    train_top2 = np.partition(train_centered, -2, axis=1)[:, -2:]
    eval_top2 = np.partition(eval_centered, -2, axis=1)[:, -2:]

    train_margin = train_top2[:, 1] - train_top2[:, 0]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]

    scale = np.median(np.abs(train_margin - np.median(train_margin))) + 1e-8
    scores = -eval_margin / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
