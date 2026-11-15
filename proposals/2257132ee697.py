import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier-selected Mahalanobis surprise in bottleneck space is sensitive to nearby held-out intents, while pooled-space surprise captures broader domain novelty. Per-predicted-intent train calibration and inverse-variance additive weighting make the two density signals comparable without using evaluation statistics."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0146"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    d_port_tr = g_port.class_mahalanobis_sq(tr.feats)[
        np.arange(len(tr.feats)), pred_tr
    ]
    d_full_tr = g_full.class_mahalanobis_sq(tr.pooled)[
        np.arange(len(tr.pooled)), pred_tr
    ]
    d_port_ev = g_port.class_mahalanobis_sq(ev.feats)[
        np.arange(len(ev.feats)), pred_ev
    ]
    d_full_ev = g_full.class_mahalanobis_sq(ev.pooled)[
        np.arange(len(ev.pooled)), pred_ev
    ]

    z_port = np.empty(len(ev.feats), dtype=float)
    z_full = np.empty(len(ev.feats), dtype=float)
    w_port = np.empty(k, dtype=float)
    w_full = np.empty(k, dtype=float)

    global_port_med = np.median(d_port_tr)
    global_full_med = np.median(d_full_tr)
    global_port_scale = np.mean(np.abs(d_port_tr - global_port_med)) + 1e-6
    global_full_scale = np.mean(np.abs(d_full_tr - global_full_med)) + 1e-6

    for c in range(k):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            a = d_port_tr[mask]
            b = d_full_tr[mask]
            ma = np.median(a)
            mb = np.median(b)
            sa = np.mean(np.abs(a - ma)) + 1e-6
            sb = np.mean(np.abs(b - mb)) + 1e-6
        else:
            ma, mb = global_port_med, global_full_med
            sa, sb = global_port_scale, global_full_scale
        idx = pred_ev == c
        z_port[idx] = (d_port_ev[idx] - ma) / sa
        z_full[idx] = (d_full_ev[idx] - mb) / sb
        w_port[c] = 1.0 / sa
        w_full[c] = 1.0 / sb

    wp = w_port[pred_ev]
    wf = w_full[pred_ev]
    return (wp * z_port + wf * z_full) / (wp + wf + 1e-12)
