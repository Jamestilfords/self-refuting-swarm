import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Known intents tend to produce a decisive separation between the best and runner-up logits, while OOS inputs often have a smaller top-two margin even when their absolute energy is misleading. Standardizing the negative margin using only training margins makes the score comparable across models."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def negative_top_two_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 0] - top2[:, 1]

    train_score = negative_top_two_margin(tr.logits)
    eval_score = negative_top_two_margin(ev.logits)

    center = np.median(train_score)
    scale = np.median(np.abs(train_score - center))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(train_score)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    return (eval_score - center) / scale
