import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The top-two logit margin captures ambiguity between competing intents that energy and "
    "maximum probability can miss. Low margins are calibrated against the training distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)

    train_top2 = np.partition(train_logits, -2, axis=1)[:, -2:]
    eval_top2 = np.partition(eval_logits, -2, axis=1)[:, -2:]
    train_margin = train_top2[:, 1] - train_top2[:, 0]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]

    order = np.sort(train_margin)
    ranks = np.searchsorted(order, eval_margin, side="right")
    score = 1.0 - ranks / float(order.size + 1)

    jitter = 1e-12 * np.tanh(np.sum(eval_logits, axis=1))
    return np.asarray(score + jitter, dtype=float)
