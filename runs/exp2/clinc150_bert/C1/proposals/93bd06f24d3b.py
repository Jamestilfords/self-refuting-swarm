import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Out-of-scope inputs often have a small gap between their two most competitive intent logits. A temperature-calibrated top-two margin uses training margins to produce a stable uncertainty score beyond energy and maximum probability."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margin(tr.logits)
    center = np.median(train_margin)
    scale = np.median(np.abs(train_margin - center))
    scale = max(float(scale), 1e-6)

    eval_margin = margin(ev.logits)
    scores = (center - eval_margin) / scale
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
