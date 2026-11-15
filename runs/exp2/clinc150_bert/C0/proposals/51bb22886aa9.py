import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This gated fusion preserves the energy ranking for low-entropy, confident predictions, while progressively replacing it with pooled-feature Mahalanobis distance when the classifier is uncertain. Feature distance can therefore rescue uncertain OOS examples whose logits alone resemble in-scope ambiguity."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)
    train_entropy = np.asarray(ctx.h.entropy(tr.logits), dtype=float)
    eval_entropy = np.asarray(ctx.h.entropy(ev.logits), dtype=float)

    def robust_standardize(train_values, values):
        lo, mid, hi = np.quantile(train_values, [0.25, 0.5, 0.75])
        scale = max(float(hi - lo), 1e-6)
        return (values - mid) / scale

    energy_z = robust_standardize(train_energy, eval_energy)
    dist_z = robust_standardize(train_dist, eval_dist)

    q25, q50, q75 = np.quantile(train_entropy, [0.25, 0.5, 0.75])
    entropy_scale = max(float(q75 - q25), 1e-6)
    gate_arg = np.clip((eval_entropy - q50) / entropy_scale, -30.0, 30.0)
    uncertainty_gate = 1.0 / (1.0 + np.exp(-gate_arg))

    scores = (1.0 - uncertainty_gate) * energy_z + uncertainty_gate * dist_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
