import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "L2-normalized bottleneck and encoder Mahalanobis distances capture complementary near-intent and far-domain novelty. Train-fitted robust tail scaling and covariance-aware positive weights prevent the more variable or redundant view from dominating."
SOURCE = "https://arxiv.org/pdf/2505.18032"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0175"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_port = ctx.h.l2(tr.feats)
    ev_port = ctx.h.l2(ev.feats)
    tr_full = ctx.h.l2(tr.pooled)
    ev_full = ctx.h.l2(ev.pooled)

    g_port = ctx.h.gda(tr_port, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr_full, tr.labels, cov="tied")

    d_port_tr = g_port.min_mahalanobis_sq(tr_port)
    d_full_tr = g_full.min_mahalanobis_sq(tr_full)
    d_port_ev = g_port.min_mahalanobis_sq(ev_port)
    d_full_ev = g_full.min_mahalanobis_sq(ev_full)

    def robust_z(train_values, values):
        med = np.median(train_values)
        mad = np.median(np.abs(train_values - med))
        scale = max(1.4826 * mad, 1e-6)
        return np.maximum((values - med) / scale, 0.0)

    z_port_tr = robust_z(d_port_tr, d_port_tr)
    z_full_tr = robust_z(d_full_tr, d_full_tr)
    z_port_ev = robust_z(d_port_tr, d_port_ev)
    z_full_ev = robust_z(d_full_tr, d_full_ev)

    train_z = np.column_stack([z_port_tr, z_full_tr])
    cov = np.cov(train_z, rowvar=False)
    cov = np.asarray(cov, dtype=float) + 1e-6 * np.eye(2)
    raw_weights = np.linalg.solve(cov, np.ones(2))
    weights = np.maximum(raw_weights, 1e-6)
    weights /= np.sum(weights)

    scores = weights[0] * z_port_ev + weights[1] * z_full_ev
    return np.asarray(scores, dtype=float)
