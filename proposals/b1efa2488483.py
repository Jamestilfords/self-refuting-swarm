import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent should be geometrically supported by its own "
    "prototype, rather than merely being close to some different intent. "
    "Training-calibrated excess distance over the best alternative class detects this disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0047", "m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_tr = ctx.h.l2(tr.pooled)
    x_ev = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(x_tr, tr.labels, cov="tied")

    d_tr = g.class_mahalanobis_sq(x_tr)
    d_ev = g.class_mahalanobis_sq(x_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    row_tr = np.arange(d_tr.shape[0])
    row_ev = np.arange(d_ev.shape[0])

    own_tr = d_tr[row_tr, pred_tr]
    own_ev = d_ev[row_ev, pred_ev]

    tmp_tr = d_tr.copy()
    tmp_ev = d_ev.copy()
    tmp_tr[row_tr, pred_tr] = np.inf
    tmp_ev[row_ev, pred_ev] = np.inf

    best_other_tr = np.min(tmp_tr, axis=1)
    best_other_ev = np.min(tmp_ev, axis=1)

    # Positive values indicate that the logit-selected class is less supported
    # than another geometric class; retain absolute own-class novelty as well.
    raw_tr = own_tr + np.maximum(own_tr - best_other_tr, 0.0)
    raw_ev = own_ev + np.maximum(own_ev - best_other_ev, 0.0)

    global_med = np.median(raw_tr)
    global_scale = np.median(np.abs(raw_tr - global_med)) * 1.4826 + 1e-6
    out = np.empty(raw_ev.shape[0], dtype=float)

    for c in range(d_tr.shape[1]):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            vals = raw_tr[mask]
            med = np.median(vals)
            scale = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
        else:
            med = global_med
            scale = global_scale
        take = pred_ev == c
        out[take] = (raw_ev[take] - med) / scale

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
