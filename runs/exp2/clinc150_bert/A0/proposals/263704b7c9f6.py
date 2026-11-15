import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query may receive a confident logit prediction while its normalized pooled representation is unusually far from that predicted intent's prototype. Class-specific train-calibrated cosine residuals expose this logit-feature disagreement without selecting the nearest class."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_tr = ctx.h.l2(tr.pooled)
    x_ev = ctx.h.l2(ev.pooled)
    labels = tr.labels.astype(int)
    k = tr.logits.shape[1]
    d = x_tr.shape[1]

    means = np.zeros((k, d), dtype=float)
    counts = np.bincount(labels, minlength=k).astype(float)
    np.add.at(means, labels, x_tr)
    means /= np.maximum(counts[:, None], 1.0)
    means = ctx.h.l2(means)

    own_cos_dist = 1.0 - np.sum(x_tr * means[labels], axis=1)
    loc = np.zeros(k, dtype=float)
    scale = np.ones(k, dtype=float)
    for c in range(k):
        vals = own_cos_dist[labels == c]
        if vals.size:
            loc[c] = np.median(vals)
            scale[c] = np.quantile(vals, 0.90) - loc[c]
    scale = np.maximum(scale, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    pred_dist = 1.0 - np.sum(x_ev * means[pred], axis=1)
    scores = (pred_dist - loc[pred]) / scale[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
