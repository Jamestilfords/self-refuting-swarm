import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope queries typically maintain a class-specific separation between their strongest and runner-up intent logits. A predicted-class standardized top-two margin flags unusually ambiguous intent assignments while accounting for naturally confusable classes."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_margin = margin(tr.logits)
    eval_margin = margin(ev.logits)

    k = tr.logits.shape[1]
    global_med = np.median(train_margin)
    global_scale = np.median(np.abs(train_margin - global_med))
    global_scale = max(1.4826 * global_scale, 1e-6)

    med = np.full(k, global_med, dtype=float)
    scale = np.full(k, global_scale, dtype=float)

    for c in range(k):
        vals = train_margin[train_pred == c]
        if vals.size >= 4:
            m = np.median(vals)
            s = np.median(np.abs(vals - m))
            med[c] = m
            scale[c] = max(1.4826 * s, 0.05 * global_scale, 1e-6)

    return (med[eval_pred] - eval_margin) / scale[eval_pred]
