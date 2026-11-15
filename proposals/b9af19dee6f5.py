import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck Mahalanobis support is sensitive to near-intent shifts, while pooled encoder "
    "support captures broader domain novelty. Predicted-class, train-calibrated normalization "
    "makes the two representation densities comparable before reliability-weighted addition."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_port_all = g_port.class_mahalanobis_sq(tr.feats)
    tr_full_all = g_full.class_mahalanobis_sq(tr.pooled)
    ev_port_all = g_port.class_mahalanobis_sq(ev.feats)
    ev_full_all = g_full.class_mahalanobis_sq(ev.pooled)

    tr_port = tr_port_all[np.arange(tr.labels.shape[0]), tr_pred]
    tr_full = tr_full_all[np.arange(tr.labels.shape[0]), tr_pred]
    ev_port = ev_port_all[np.arange(ev_pred.shape[0]), ev_pred]
    ev_full = ev_full_all[np.arange(ev_pred.shape[0]), ev_pred]

    k = tr.logits.shape[1]
    global_port_mean = float(np.mean(tr_port))
    global_full_mean = float(np.mean(tr_full))
    global_port_std = max(float(np.std(tr_port)), 1e-6)
    global_full_std = max(float(np.std(tr_full)), 1e-6)

    port_mean = np.full(k, global_port_mean)
    full_mean = np.full(k, global_full_mean)
    port_std = np.full(k, global_port_std)
    full_std = np.full(k, global_full_std)

    for c in range(k):
        mask = tr_pred == c
        if np.sum(mask) >= 8:
            port_mean[c] = np.mean(tr_port[mask])
            full_mean[c] = np.mean(tr_full[mask])
            port_std[c] = max(float(np.std(tr_port[mask])), 1e-6)
            full_std[c] = max(float(np.std(tr_full[mask])), 1e-6)

    z_port_train = (tr_port - port_mean[tr_pred]) / port_std[tr_pred]
    z_full_train = (tr_full - full_mean[tr_pred]) / full_std[tr_pred]

    reliability_port = 1.0 / max(float(np.std(z_port_train)), 1e-6)
    reliability_full = 1.0 / max(float(np.std(z_full_train)), 1e-6)
    total = reliability_port + reliability_full
    w_port = reliability_port / total
    w_full = reliability_full / total

    z_port_eval = (ev_port - port_mean[ev_pred]) / port_std[ev_pred]
    z_full_eval = (ev_full - full_mean[ev_pred]) / full_std[ev_pred]

    return w_port * z_port_eval + w_full * z_full_eval
