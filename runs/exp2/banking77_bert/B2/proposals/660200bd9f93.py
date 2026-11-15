import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck-space density captures local support around known intents, while encoder-space "
    "density captures broader semantic and domain shifts. Train-fitted inverse-dispersion weights "
    "place both distances on comparable scales before additive fusion."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = np.asarray(g_port.min_mahalanobis_sq(tr.feats), dtype=float)
    full_train = np.asarray(g_full.min_mahalanobis_sq(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.min_mahalanobis_sq(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.min_mahalanobis_sq(ev.pooled), dtype=float)

    def center_scale(x):
        center = np.median(x)
        scale = np.median(np.abs(x - center)) * 1.4826
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(x)
        return center, max(float(scale), 1e-8)

    port_center, port_scale = center_scale(port_train)
    full_center, full_scale = center_scale(full_train)

    inv_port = 1.0 / port_scale
    inv_full = 1.0 / full_scale
    total = inv_port + inv_full
    w_port = inv_port / total
    w_full = inv_full / total

    port_z = (port_eval - port_center) / port_scale
    full_z = (full_eval - full_center) / full_scale
    scores = w_port * port_z + w_full * full_z

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
