import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should lie at a typical feature-space radius from the mean of the intent "
    "chosen by the classifier. Robust, predicted-intent-specific calibration exposes cases where "
    "the logit decision is unsupported by that intent's representation cluster."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    ytr = np.asarray(tr.labels, dtype=int)

    k = tr.logits.shape[1]
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(xtr, axis=0)
    means = np.empty((k, xtr.shape[1]), dtype=float)
    for c in range(k):
        members = xtr[ytr == c]
        means[c] = np.mean(members, axis=0) if len(members) else global_mean

    tr_dist = np.sqrt(np.sum((xtr - means[pred_tr]) ** 2, axis=1))
    ev_dist = np.sqrt(np.sum((xev - means[pred_ev]) ** 2, axis=1))

    med = np.empty(k, dtype=float)
    scale = np.empty(k, dtype=float)
    fallback_med = np.median(tr_dist)
    fallback_scale = np.median(np.abs(tr_dist - fallback_med))
    fallback_scale = max(float(fallback_scale), 1e-6)

    for c in range(k):
        vals = tr_dist[pred_tr == c]
        if len(vals) >= 4:
            m = np.median(vals)
            s = np.median(np.abs(vals - m))
            med[c] = m
            scale[c] = max(float(s), 1e-6)
        else:
            med[c] = fallback_med
            scale[c] = fallback_scale

    return (ev_dist - med[pred_ev]) / scale[pred_ev]
