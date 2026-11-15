import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy retains control for train-calibrated confident predictions, preserving its useful "
    "ranking there, while a pooled-space Mahalanobis score progressively replaces it for uncertain "
    "predictions where representation support is more informative."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0077", "m_0081"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)

    def robust_location_scale(x):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        med = float(np.median(x))
        q25, q75 = np.percentile(x, [25.0, 75.0])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = max(float(np.mean(np.abs(x - med))), 1.0)
        return med, scale

    e_med, e_scale = robust_location_scale(e_tr)
    d_med, d_scale = robust_location_scale(d_tr)

    # The upper-middle train energy range marks the transition from confident
    # classifier behavior to the uncertainty regime where feature support matters.
    transition = float(np.percentile(np.nan_to_num(e_tr, nan=e_med), 60.0))
    gate_scale = max(e_scale * 0.45, 1e-6)

    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)
    e_ev = np.nan_to_num(e_ev, nan=e_med, posinf=e_med + 50.0 * e_scale,
                         neginf=e_med - 50.0 * e_scale)
    d_ev = np.nan_to_num(d_ev, nan=d_med, posinf=d_med + 50.0 * d_scale,
                         neginf=d_med - 50.0 * d_scale)

    ez = np.clip((e_ev - e_med) / e_scale, -50.0, 50.0)
    dz = np.clip((d_ev - d_med) / d_scale, -50.0, 50.0)

    gate_arg = np.clip((e_ev - transition) / gate_scale, -50.0, 50.0)
    uncertainty_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    return (1.0 - uncertainty_gate) * ez + uncertainty_gate * dz
