import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low energy indicates confident intent assignments, so the score preserves energy ordering there. As energy becomes atypical relative to training data, it smoothly hands off to pooled-feature Mahalanobis distance, which can expose geometrically unsupported near-OOS assignments."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0122", "m_0139"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    ev_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    tr_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    ev_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    def robust_z(x, ref):
        med = np.median(ref)
        scale = np.percentile(ref, 75) - np.percentile(ref, 25)
        scale = max(float(scale), 1e-6)
        return (x - med) / scale

    e_train = robust_z(tr_energy, tr_energy)
    e_eval = robust_z(ev_energy, tr_energy)
    d_eval = robust_z(ev_dist, tr_dist)

    handoff_center = np.percentile(e_train, 60)
    handoff_width = max(
        float(np.percentile(e_train, 75) - np.percentile(e_train, 25)),
        1e-3,
    )
    gate = 1.0 / (1.0 + np.exp(-np.clip(
        (e_eval - handoff_center) / handoff_width, -30.0, 30.0
    )))

    out = (1.0 - gate) * e_eval + gate * d_eval
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
