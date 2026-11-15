import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and pooled encoder distances capture near-intent and broader domain shifts, but their support scales can vary substantially by classifier-selected intent. Predicted-intent robust calibration followed by train-fitted covariance-aware additive weighting prevents one redundant view from dominating the fused anomaly signal."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0120", "m_0134"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_classes = tr.logits.shape[1]

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_tr_all = np.asarray(g_port.class_mahalanobis_sq(tr.feats), dtype=float)
    full_tr_all = np.asarray(g_full.class_mahalanobis_sq(tr.pooled), dtype=float)
    port_ev_all = np.asarray(g_port.class_mahalanobis_sq(ev.feats), dtype=float)
    full_ev_all = np.asarray(g_full.class_mahalanobis_sq(ev.pooled), dtype=float)

    port_tr = port_tr_all[np.arange(port_tr_all.shape[0]), pred_tr]
    full_tr = full_tr_all[np.arange(full_tr_all.shape[0]), pred_tr]
    port_ev = port_ev_all[np.arange(port_ev_all.shape[0]), pred_ev]
    full_ev = full_ev_all[np.arange(full_ev_all.shape[0]), pred_ev]

    port_tr = np.log1p(np.maximum(port_tr, 0.0))
    full_tr = np.log1p(np.maximum(full_tr, 0.0))
    port_ev = np.log1p(np.maximum(port_ev, 0.0))
    full_ev = np.log1p(np.maximum(full_ev, 0.0))

    global_port_med = float(np.median(port_tr))
    global_full_med = float(np.median(full_tr))
    global_port_iqr = float(np.percentile(port_tr, 75) - np.percentile(port_tr, 25))
    global_full_iqr = float(np.percentile(full_tr, 75) - np.percentile(full_tr, 25))
    global_port_iqr = max(global_port_iqr, 1e-6)
    global_full_iqr = max(global_full_iqr, 1e-6)

    med_port = np.full(n_classes, global_port_med, dtype=float)
    med_full = np.full(n_classes, global_full_med, dtype=float)
    iqr_port = np.full(n_classes, global_port_iqr, dtype=float)
    iqr_full = np.full(n_classes, global_full_iqr, dtype=float)

    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            a = port_tr[mask]
            b = full_tr[mask]
            med_port[c] = np.median(a)
            med_full[c] = np.median(b)
            iqr_port[c] = max(float(np.percentile(a, 75) - np.percentile(a, 25)), 1e-6)
            iqr_full[c] = max(float(np.percentile(b, 75) - np.percentile(b, 25)), 1e-6)

    z_port_tr = (port_tr - med_port[pred_tr]) / iqr_port[pred_tr]
    z_full_tr = (full_tr - med_full[pred_tr]) / iqr_full[pred_tr]
    z_port_ev = (port_ev - med_port[pred_ev]) / iqr_port[pred_ev]
    z_full_ev = (full_ev - med_full[pred_ev]) / iqr_full[pred_ev]

    train_pair = np.column_stack((z_port_tr, z_full_tr))
    cov = np.cov(train_pair, rowvar=False)
    cov = np.asarray(cov, dtype=float) + 1e-5 * np.eye(2)
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(2, dtype=float)
    weights = inv_cov @ ones
    weights = np.maximum(weights, 0.05)
    weights /= np.sum(weights)

    scores = weights[0] * z_port_ev + weights[1] * z_full_ev
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
