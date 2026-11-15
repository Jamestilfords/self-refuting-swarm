import numpy as np
from scipy.special import ndtri

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier energy and pooled-space Mahalanobis novelty are calibrated separately within the classifier's predicted intent, so naturally difficult intents do not dominate either signal. Averaging their train-calibrated Gaussian quantiles flags queries that are unusually uncertain or geometrically unsupported for the intent they resemble."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0099"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    sorted_energy = {}
    sorted_dist = {}
    global_energy = np.sort(train_energy)
    global_dist = np.sort(train_dist)

    for c in np.unique(train_pred):
        mask = train_pred == c
        if np.sum(mask) >= 8:
            sorted_energy[int(c)] = np.sort(train_energy[mask])
            sorted_dist[int(c)] = np.sort(train_dist[mask])

    def calibrated(values, preds, tables, fallback):
        out = np.empty(values.shape[0], dtype=float)
        for c in np.unique(preds):
            mask = preds == c
            ref = tables.get(int(c), fallback)
            ranks = np.searchsorted(ref, values[mask], side="right")
            out[mask] = (ranks + 0.5) / (ref.size + 1.0)
        return out

    q_energy = calibrated(eval_energy, eval_pred, sorted_energy, global_energy)
    q_dist = calibrated(eval_dist, eval_pred, sorted_dist, global_dist)

    q_energy = np.clip(q_energy, 1e-5, 1.0 - 1e-5)
    q_dist = np.clip(q_dist, 1e-5, 1.0 - 1e-5)
    return 0.5 * (ndtri(q_energy) + ndtri(q_dist))
