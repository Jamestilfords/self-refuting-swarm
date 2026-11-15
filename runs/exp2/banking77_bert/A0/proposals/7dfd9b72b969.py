import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query should be angularly compatible with the prototype of the class predicted by the classifier, "
    "and that compatibility should be unusually strong relative to competing intent prototypes. "
    "Class-conditional train calibration makes low predicted-prototype cosine and weak angular separation high OOS scores."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_tr = ctx.h.l2(tr.feats)
    x_ev = ctx.h.l2(ev.feats)
    y = tr.labels.astype(int)
    k = tr.logits.shape[1]

    prototypes = np.zeros((k, x_tr.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(prototypes, y, x_tr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_cos = x_tr @ prototypes.T
    ev_cos = x_ev @ prototypes.T

    tr_own = tr_cos[np.arange(tr_cos.shape[0]), tr_pred]
    ev_own = ev_cos[np.arange(ev_cos.shape[0]), ev_pred]

    tr_masked = tr_cos.copy()
    tr_masked[np.arange(tr_cos.shape[0]), tr_pred] = -np.inf
    ev_masked = ev_cos.copy()
    ev_masked[np.arange(ev_cos.shape[0]), ev_pred] = -np.inf

    tr_margin = tr_own - np.max(tr_masked, axis=1)
    ev_margin = ev_own - np.max(ev_masked, axis=1)

    global_own_mean = float(np.mean(tr_own))
    global_margin_mean = float(np.mean(tr_margin))
    global_own_std = max(float(np.std(tr_own)), 1e-6)
    global_margin_std = max(float(np.std(tr_margin)), 1e-6)

    own_mean = np.full(k, global_own_mean, dtype=float)
    own_std = np.full(k, global_own_std, dtype=float)
    margin_mean = np.full(k, global_margin_mean, dtype=float)
    margin_std = np.full(k, global_margin_std, dtype=float)

    for c in range(k):
        mask = tr_pred == c
        if np.sum(mask) >= 3:
            own_mean[c] = np.mean(tr_own[mask])
            own_std[c] = max(float(np.std(tr_own[mask])), 1e-6)
            margin_mean[c] = np.mean(tr_margin[mask])
            margin_std[c] = max(float(np.std(tr_margin[mask])), 1e-6)

    own_deficit = (own_mean[ev_pred] - ev_own) / own_std[ev_pred]
    margin_deficit = (margin_mean[ev_pred] - ev_margin) / margin_std[ev_pred]

    return 0.7 * own_deficit + 0.3 * margin_deficit
