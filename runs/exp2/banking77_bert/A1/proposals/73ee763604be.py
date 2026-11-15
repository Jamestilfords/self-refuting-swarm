import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query can be confidently assigned to an intent while lying unusually far from that predicted "
    "intent's angular prototype. Class-conditional robust calibration accounts for intents with different "
    "within-class semantic spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.pooled)
    xev = ctx.h.l2(ev.pooled)
    labels = tr.labels.astype(int)
    k = tr.logits.shape[1]

    global_mean = np.mean(xtr, axis=0)
    global_mean = global_mean / max(np.linalg.norm(global_mean), 1e-12)
    means = np.tile(global_mean, (k, 1))

    for c in range(k):
        mask = labels == c
        if np.any(mask):
            mu = np.mean(xtr[mask], axis=0)
            means[c] = mu / max(np.linalg.norm(mu), 1e-12)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    tr_dist = 1.0 - np.sum(xtr * means[pred_tr], axis=1)
    ev_dist = 1.0 - np.sum(xev * means[pred_ev], axis=1)

    global_med = np.median(tr_dist)
    global_iqr = np.percentile(tr_dist, 75) - np.percentile(tr_dist, 25)
    global_scale = max(global_iqr, 1e-6)

    med = np.full(k, global_med)
    scale = np.full(k, global_scale)
    for c in range(k):
        vals = tr_dist[pred_tr == c]
        if vals.size >= 8:
            med[c] = np.median(vals)
            iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
            scale[c] = max(iqr, 1e-6)

    return (ev_dist - med[pred_ev]) / scale[pred_ev]
