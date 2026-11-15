import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to produce class-specific logit shapes with larger centered logit norms, while OOS inputs often yield flatter, lower-contrast logits. A class-calibrated deficit in centered logit norm captures this shape information without reducing to energy or maximum probability."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_centered = train_logits - np.mean(train_logits, axis=1, keepdims=True)
    eval_centered = eval_logits - np.mean(eval_logits, axis=1, keepdims=True)

    train_norm = np.sqrt(np.sum(train_centered * train_centered, axis=1))
    eval_norm = np.sqrt(np.sum(eval_centered * eval_centered, axis=1))

    k = train_logits.shape[1]
    global_med = np.median(train_norm)
    global_mad = np.median(np.abs(train_norm - global_med)) + 1e-6

    med = np.full(k, global_med, dtype=float)
    mad = np.full(k, global_mad, dtype=float)

    for c in range(k):
        vals = train_norm[labels == c]
        if vals.size:
            med[c] = np.median(vals)
            mad[c] = np.median(np.abs(vals - med[c])) + 1e-6

    pred = np.argmax(eval_logits, axis=1)
    norm_deficit = (med[pred] - eval_norm) / mad[pred]

    top2 = np.partition(eval_logits, -2, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    return norm_deficit - 0.05 * margin
