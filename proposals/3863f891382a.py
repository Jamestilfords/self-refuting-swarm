FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A query whose classifier-selected intent is geometrically much farther than the nearest "
    "feature-space intent exhibits classifier-feature disagreement. Class-conditional train "
    "standardization also retains unusually large absolute distance to the selected intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0018"]


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    d_tr_all = g.class_mahalanobis_sq(tr.pooled)
    d_ev_all = g.class_mahalanobis_sq(ev.pooled)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    row_tr = np.arange(tr.logits.shape[0])
    row_ev = np.arange(ev.logits.shape[0])

    dsel_tr = d_tr_all[row_tr, pred_tr]
    dsel_ev = d_ev_all[row_ev, pred_ev]

    dnear_tr = np.min(d_tr_all, axis=1)
    dnear_ev = np.min(d_ev_all, axis=1)

    # Positive excess is zero when the classifier-selected class is also
    # the geometrically closest known class.
    excess_tr = np.maximum(dsel_tr - dnear_tr, 0.0)
    excess_ev = np.maximum(dsel_ev - dnear_ev, 0.0)

    n_classes = tr.logits.shape[1]
    global_d_mean = float(np.mean(dsel_tr))
    global_d_scale = float(np.std(dsel_tr))
    global_e_mean = float(np.mean(excess_tr))
    global_e_scale = float(np.std(excess_tr))

    if global_d_scale < 1e-6:
        global_d_scale = 1.0
    if global_e_scale < 1e-6:
        global_e_scale = 1.0

    d_mean = np.full(n_classes, global_d_mean)
    d_scale = np.full(n_classes, global_d_scale)
    e_mean = np.full(n_classes, global_e_mean)
    e_scale = np.full(n_classes, global_e_scale)

    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) >= 4:
            dm = float(np.mean(dsel_tr[mask]))
            ds = float(np.std(dsel_tr[mask]))
            em = float(np.mean(excess_tr[mask]))
            es = float(np.std(excess_tr[mask]))
            d_mean[c] = dm
            d_scale[c] = max(ds, 1e-6)
            e_mean[c] = em
            e_scale[c] = max(es, 1e-6)

    selected_distance_z = (dsel_ev - d_mean[pred_ev]) / d_scale[pred_ev]
    disagreement_z = (excess_ev - e_mean[pred_ev]) / e_scale[pred_ev]

    out = selected_distance_z + 0.75 * disagreement_z
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
