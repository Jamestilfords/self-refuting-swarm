import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This gate preserves energy ranking for training-like confident queries, while progressively relying on class-conditional feature distance when energy enters its uncertain upper tail."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    e_train = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    d_train = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    e_eval = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_eval = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_med = np.median(e_train)
    d_med = np.median(d_train)
    e_iqr = np.percentile(e_train, 75) - np.percentile(e_train, 25)
    d_iqr = np.percentile(d_train, 75) - np.percentile(d_train, 25)

    e_scale = max(float(e_iqr), 1e-6)
    d_scale = max(float(d_iqr), 1e-6)
    uncertainty_start = np.percentile(e_train, 75)

    e_norm = (e_eval - e_med) / e_scale
    d_norm = (d_eval - d_med) / d_scale

    gate_arg = np.clip((e_eval - uncertainty_start) / e_scale, -30.0, 30.0)
    distance_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    scores = (1.0 - distance_gate) * e_norm + distance_gate * d_norm
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
