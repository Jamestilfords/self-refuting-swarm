import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains unchanged for training-calibrated confident predictions, while a smooth uncertainty gate increasingly substitutes class-selected pooled-feature support. Class-conditional distance tails make feature novelty comparable across intents only when logit evidence is weak."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0246"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    # A confident item is scored exactly by energy.  The transition width is
    # fitted entirely from the training-energy distribution.
    e_start = float(np.quantile(e_tr, 0.70))
    e_end = float(np.quantile(e_tr, 0.90))
    width = max(e_end - e_start, 1e-6)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    d_tr_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    d_tr = d_tr_all[np.arange(len(pred_tr)), pred_tr]
    d_ev = d_ev_all[np.arange(len(pred_ev)), pred_ev]

    # Fit a separate empirical support tail for the classifier-selected
    # intent.  A global fallback handles a rarely predicted intent.
    global_sorted = np.sort(d_tr)
    sorted_by_class = []
    for c in range(tr.logits.shape[1]):
        vals = d_tr[pred_tr == c]
        sorted_by_class.append(np.sort(vals) if len(vals) >= 3 else global_sorted)

    tail = np.empty(len(pred_ev), dtype=float)
    for c in range(tr.logits.shape[1]):
        mask = pred_ev == c
        if np.any(mask):
            ref = sorted_by_class[c]
            rank = np.searchsorted(ref, d_ev[mask], side="right")
            cdf = rank / float(len(ref) + 1)
            # Negative log survival emphasizes points beyond normal support
            # while remaining finite for arbitrary evaluation distances.
            tail[mask] = -np.log1p(-cdf)

    gate = np.clip((e_ev - e_start) / width, 0.0, 1.0)
    energy_excess = (e_ev - e_start) / width

    # Below e_start this equals e_ev exactly.  Above it, feature-tail support
    # progressively becomes the dominant ranking signal.
    return e_start + width * ((1.0 - gate) * energy_excess + gate * tail)
