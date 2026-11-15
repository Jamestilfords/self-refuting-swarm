import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis distance is sensitive to near-OOS held-out intents, while encoder-space distance captures broader domain shifts; train-fitted robust scaling makes their additive fusion comparable."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = np.asarray(g_port.min_mahalanobis_sq(tr.feats), dtype=float)
    full_train = np.asarray(g_full.min_mahalanobis_sq(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.min_mahalanobis_sq(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_scale(train_values, eval_values):
        lo, hi = np.quantile(train_values, [0.10, 0.90])
        scale = max(float(hi - lo), 1e-8)
        return (eval_values - lo) / scale, scale

    z_port, s_port = robust_scale(port_train, port_eval)
    z_full, s_full = robust_scale(full_train, full_eval)

    # Inverse train-space spread weights prevent the higher-dimensional
    # encoder density from dominating solely because of its raw scale.
    inv = np.array([1.0 / s_port, 1.0 / s_full])
    weights = inv / np.sum(inv)

    scores = weights[0] * z_port + weights[1] * z_full
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
