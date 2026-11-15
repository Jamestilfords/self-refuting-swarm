import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the score for train-calibrated confident queries, while a smooth high-energy gate transfers weight to predicted-intent bottleneck Mahalanobis surprise when classifier evidence is uncertain. This targets examples whose uncertain routing is better resolved by representation compatibility."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0105", "m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_all_d = g.class_mahalanobis_sq(tr.feats)
    ev_all_d = g.class_mahalanobis_sq(ev.feats)
    tr_d = tr_all_d[np.arange(tr.feats.shape[0]), tr_pred]
    ev_d = ev_all_d[np.arange(ev.feats.shape[0]), ev_pred]

    tr_energy = ctx.h.energy(tr.logits)
    ev_energy = ctx.h.energy(ev.logits)
    energy_ref = np.sort(np.asarray(tr_energy, dtype=float))
    energy_pct = np.searchsorted(energy_ref, ev_energy, side="right") / float(len(energy_ref))

    global_ref = np.sort(np.asarray(tr_d, dtype=float))
    dist_pct = np.empty(ev_d.shape[0], dtype=float)

    for c in range(tr_all_d.shape[1]):
        train_c = tr_d[tr_pred == c]
        mask = ev_pred == c
        if train_c.size >= 8:
            ref = np.sort(train_c)
        else:
            ref = global_ref
        dist_pct[mask] = np.searchsorted(ref, ev_d[mask], side="right") / float(len(ref))

    gate = 1.0 / (1.0 + np.exp(-(energy_pct - 0.65) / 0.08))
    return (1.0 - gate) * energy_pct + gate * dist_pct
