import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin captures ambiguity between the two most plausible intents, "
    "which can flag OOS queries even when total logit scale makes energy less informative."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_sorted = np.sort(tr.logits, axis=1)
    margins = train_sorted[:, -1] - train_sorted[:, -2]
    scale = np.median(margins)
    scale = max(float(scale), 1e-6)

    eval_sorted = np.sort(ev.logits, axis=1)
    eval_margins = eval_sorted[:, -1] - eval_sorted[:, -2]
    scores = -eval_margins / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
