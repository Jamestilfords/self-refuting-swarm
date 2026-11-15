import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two logit margin measures decisional separation not captured by energy alone. Class-conditional lower-tail calibration flags queries whose predicted intent is supported by an unusually ambiguous logit ordering."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin_and_pred(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        pred = np.argmax(logits, axis=1)
        return margin, pred

    train_margin, train_pred = margin_and_pred(tr.logits)
    eval_margin, eval_pred = margin_and_pred(ev.logits)

    global_sorted = np.sort(train_margin)
    out = np.empty(len(eval_margin), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        values = np.sort(train_margin[mask])
        if values.size < 8:
            values = global_sorted
        where = eval_pred == c
        ranks = np.searchsorted(values, eval_margin[where], side="right")
        out[where] = 1.0 - (ranks + 0.5) / (values.size + 1.0)

    return out
