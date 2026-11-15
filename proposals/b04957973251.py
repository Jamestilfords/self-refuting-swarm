import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier-selected class densities in bottleneck and pooled encoder spaces capture "
    "near-intent and far-domain shifts respectively. Per-predicted-intent robust calibration "
    "and train-fitted minimum-variance weights make their additive surprise signals comparable."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0295", "m_0305"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    d_port_tr_all = g_port.class_mahalanobis_sq(tr.feats)
    d_full_tr_all = g_full.class_mahalanobis_sq(tr.pooled)
    d_port_ev_all = g_port.class_mahalanobis_sq(ev.feats)
    d_full_ev_all = g_full.class_mahalanobis_sq(ev.pooled)

    idx_tr = np.arange(tr.logits.shape[0])
    idx_ev = np.arange(ev.logits.shape[0])

    d_port_tr = d_port_tr_all[idx_tr, pred_tr]
    d_full_tr = d_full_tr_all[idx_tr, pred_tr]
    d_port_ev = d_port_ev_all[idx_ev, pred_ev]
    d_full_ev = d_full_ev_all[idx_ev, pred_ev]

    k = tr.logits.shape[1]
    med_port = np.empty(k, dtype=float)
    scl_port = np.empty(k, dtype=float)
    med_full = np.empty(k, dtype=float)
    scl_full = np.empty(k, dtype=float)

    global_med_port = np.median(d_port_tr)
    global_med_full = np.median(d_full_tr)
    global_scl_port = np.percentile(d_port_tr, 75) - np.percentile(d_port_tr, 25)
    global_scl_full = np.percentile(d_full_tr, 75) - np.percentile(d_full_tr, 25)

    global_scl_port = max(float(global_scl_port), 1e-6)
    global_scl_full = max(float(global_scl_full), 1e-6)

    for c in range(k):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            a = d_port_tr[mask]
            b = d_full_tr[mask]
            med_port[c] = np.median(a)
            med_full[c] = np.median(b)
            scl_port[c] = np.percentile(a, 75) - np.percentile(a, 25)
            scl_full[c] = np.percentile(b, 75) - np.percentile(b, 25)
        else:
            med_port[c] = global_med_port
            med_full[c] = global_med_full
            scl_port[c] = global_scl_port
            scl_full[c] = global_scl_full

    scl_port = np.maximum(scl_port, 1e-6)
    scl_full = np.maximum(scl_full, 1e-6)

    z_port_tr = (d_port_tr - med_port[pred_tr]) / scl_port[pred_tr]
    z_full_tr = (d_full_tr - med_full[pred_tr]) / scl_full[pred_tr]
    z_port_ev = (d_port_ev - med_port[pred_ev]) / scl_port[pred_ev]
    z_full_ev = (d_full_ev - med_full[pred_ev]) / scl_full[pred_ev]

    s_port_tr = np.logaddexp(0.0, z_port_tr)
    s_full_tr = np.logaddexp(0.0, z_full_tr)
    s_port_ev = np.logaddexp(0.0, z_port_ev)
    s_full_ev = np.logaddexp(0.0, z_full_ev)

    train_scores = np.column_stack((s_port_tr, s_full_tr))
    cov = np.cov(train_scores, rowvar=False)
    cov = np.asarray(cov, dtype=float) + 1e-5 * np.eye(2)

    ones = np.ones(2, dtype=float)
    weights = np.linalg.solve(cov, ones)
    weights = np.maximum(weights, 1e-6)
    weights = weights / np.sum(weights)

    out = weights[0] * s_port_ev + weights[1] * s_full_ev
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
