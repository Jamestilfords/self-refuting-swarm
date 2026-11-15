import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck Mahalanobis distance is sensitive to nearby held-out intent shifts, while pooled "
    "encoder distance captures broader domain novelty. Train-fitted robust tail scales make their "
    "additive fusion comparable without using evaluation statistics."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0273", "m_0159"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_port = g_port.min_mahalanobis_sq(tr.feats)
    train_full = g_full.min_mahalanobis_sq(tr.pooled)
    eval_port = g_port.min_mahalanobis_sq(ev.feats)
    eval_full = g_full.min_mahalanobis_sq(ev.pooled)

    def robust_tail_scale(x):
        q25, q50, q75, q90 = np.percentile(x, [25.0, 50.0, 75.0, 90.0])
        scale = max(q75 - q25, 1e-6)
        tail = max(q90 - q50, scale)
        return q50, tail

    med_port, scale_port = robust_tail_scale(train_port)
    med_full, scale_full = robust_tail_scale(train_full)

    z_port_train = np.maximum((train_port - med_port) / scale_port, 0.0)
    z_full_train = np.maximum((train_full - med_full) / scale_full, 0.0)

    rel_port = 1.0 / max(np.mean(z_port_train * z_port_train), 1e-6)
    rel_full = 1.0 / max(np.mean(z_full_train * z_full_train), 1e-6)
    total = rel_port + rel_full
    w_port = rel_port / total
    w_full = rel_full / total

    z_port = np.maximum((eval_port - med_port) / scale_port, 0.0)
    z_full = np.maximum((eval_full - med_full) / scale_full, 0.0)
    return w_port * z_port + w_full * z_full
