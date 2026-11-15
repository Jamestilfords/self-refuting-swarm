import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin measures decisiveness among the most plausible intents, "
    "while train-calibrated logit norm discounts uniformly weak but accidentally separated predictions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_value(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - logits.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.mean(centered * centered, axis=1))
        return -(margin * norm)

    train_score = shape_value(tr.logits)
    eval_score = shape_value(ev.logits)

    lo, hi = np.quantile(train_score, [0.01, 0.99])
    scale = max(hi - lo, 1e-8)
    return np.clip((eval_score - lo) / scale, -10.0, 10.0)
