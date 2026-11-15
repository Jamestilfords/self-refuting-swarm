import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident predictions retain their raw energy ordering, while only examples in a "
    "class-conditioned uncertain-energy regime receive a pooled-feature support penalty."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0215", "m_0051"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    d_tr_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    d_tr = d_tr_all[np.arange(len(tr.labels)), pred_tr]
    d_ev = d_ev_all[np.arange(len(pred_ev)), pred_ev]

    k = tr.logits.shape[1]
    thresholds = np.full(k, np.quantile(e_tr, 0.70), dtype=float)
    penalties = np.zeros(len(pred_ev), dtype=float)

    for c in range(k):
        train_mask = pred_tr == c
        eval_mask = pred_ev == c
        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) >= 8:
            class_energy = e_tr[train_mask]
            class_dist = d_tr[train_mask]
            threshold = np.quantile(class_energy, 0.70)
            uncertain_dist = class_dist[class_energy >= threshold]
            if len(uncertain_dist) < 4:
                uncertain_dist = class_dist
        else:
            threshold = thresholds[c]
            uncertain_dist = d_tr

        thresholds[c] = threshold
        reference = np.sort(np.nan_to_num(
            uncertain_dist,
            nan=np.nanmedian(d_tr),
            posinf=np.nanmax(np.nan_to_num(d_tr, nan=0.0, posinf=0.0)),
            neginf=0.0,
        ))
        reference = reference[np.isfinite(reference)]
        if len(reference) == 0:
            reference = np.array([0.0])

        values = np.nan_to_num(d_ev[eval_mask], nan=reference[-1],
                               posinf=reference[-1], neginf=reference[0])
        ranks = np.searchsorted(reference, values, side="right") / float(len(reference))
        penalties[eval_mask] = ranks

    uncertainty = np.maximum(e_ev - thresholds[pred_ev], 0.0)
    scale = np.median(np.abs(e_tr - np.median(e_tr))) + 1e-6
    gate = 1.0 - np.exp(-uncertainty / scale)
    return np.nan_to_num(e_ev + gate * penalties, nan=0.0, posinf=1e12, neginf=-1e12)
