import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Class-conditional energy tails route only uncertain predictions toward normalized pooled-feature "
    "Mahalanobis novelty, while confident items retain their original energy ordering."
)
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_lo, global_hi = np.quantile(train_energy, [0.70, 0.95])
    if global_hi <= global_lo:
        global_hi = global_lo + 1.0

    class_lo = np.full(n_classes, global_lo, dtype=float)
    class_hi = np.full(n_classes, global_hi, dtype=float)
    for c in range(n_classes):
        values = train_energy[train_pred == c]
        if values.size >= 12:
            lo, hi = np.quantile(values, [0.70, 0.95])
            if hi > lo:
                class_lo[c] = lo
                class_hi[c] = hi

    tr_x = ctx.h.l2(tr.pooled)
    ev_x = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(tr_x, tr.labels, cov="tied", whiten=True)

    train_dist = np.asarray(g.min_mahalanobis_sq(tr_x), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev_x), dtype=float)

    d_lo, d_hi = np.quantile(train_dist, [0.05, 0.95])
    if d_hi <= d_lo:
        d_hi = d_lo + 1.0

    novelty = np.clip((eval_dist - d_lo) / (d_hi - d_lo), 0.0, 1.0)
    lo = class_lo[eval_pred]
    hi = class_hi[eval_pred]

    gate = np.clip((eval_energy - lo) / (hi - lo), 0.0, 1.0)
    feature_equivalent = lo + (hi - lo) * novelty
    scores = (1.0 - gate) * eval_energy + gate * feature_equivalent

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
