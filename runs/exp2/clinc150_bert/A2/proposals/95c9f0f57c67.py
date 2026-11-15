import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two logit margin measures how decisively the classifier separates its selected intent from its nearest alternative, independently of the common logit offset captured by energy. OOS queries may retain high absolute logits yet show unusually weak relative intent separation."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margin(tr.logits)
    eval_margin = margin(ev.logits)

    order = np.sort(train_margin)
    n = order.size
    ranks = np.searchsorted(order, eval_margin, side="right")
    score_values = 1.0 - (ranks + 0.5) / (n + 1.0)

    if np.ptp(score_values) == 0.0:
        return -eval_margin
    return score_values
