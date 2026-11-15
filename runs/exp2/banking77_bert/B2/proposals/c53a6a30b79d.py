import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope queries tend to have a decisive gap between their most and second-most supported intents. "
    "A small top-two logit margin flags ambiguous intent allocation beyond total logit energy or maximum probability."
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

    center = np.median(train_margin)
    scale = np.median(np.abs(train_margin - center)) + 1e-8
    scores = (center - eval_margin) / scale
    return np.asarray(scores, dtype=float)
