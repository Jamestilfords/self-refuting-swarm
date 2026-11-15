import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should be angularly close to their classifier-selected prototype while also being more aligned with it than with competing intent prototypes. A class-calibrated deficit in both prototype support and angular separation flags geometrically ambiguous OOS queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0036", "m_0075", "m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    y = tr.labels.astype(int)
    k = tr.logits.shape[1]

    def normalize(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    xtr = normalize(tr.feats.astype(float))
    xev = normalize(ev.feats.astype(float))

    sums = np.zeros((k, xtr.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=k).astype(float)
    np.add.at(sums, y, xtr)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = normalize(prototypes)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    sim_tr = xtr @ prototypes.T
    sim_ev = xev @ prototypes.T

    own_tr = sim_tr[np.arange(len(xtr)), pred_tr]
    own_ev = sim_ev[np.arange(len(xev)), pred_ev]

    alt_tr = sim_tr.copy()
    alt_tr[np.arange(len(xtr)), pred_tr] = -np.inf
    alt_ev = sim_ev.copy()
    alt_ev[np.arange(len(xev)), pred_ev] = -np.inf

    margin_tr = own_tr - np.max(alt_tr, axis=1)
    margin_ev = own_ev - np.max(alt_ev, axis=1)

    def class_stats(values):
        total = np.zeros(k, dtype=float)
        total_sq = np.zeros(k, dtype=float)
        pred_counts = np.bincount(pred_tr, minlength=k).astype(float)
        np.add.at(total, pred_tr, values)
        np.add.at(total_sq, pred_tr, values * values)
        mean = total / np.maximum(pred_counts, 1.0)
        var = total_sq / np.maximum(pred_counts, 1.0) - mean * mean
        return mean, np.sqrt(np.maximum(var, 1e-6))

    own_mean, own_std = class_stats(own_tr)
    margin_mean, margin_std = class_stats(margin_tr)

    support_deficit = (own_mean[pred_ev] - own_ev) / own_std[pred_ev]
    margin_deficit = (margin_mean[pred_ev] - margin_ev) / margin_std[pred_ev]

    return 0.55 * support_deficit + 0.45 * margin_deficit
