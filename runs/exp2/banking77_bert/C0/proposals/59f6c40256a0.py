import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This confidence-gated fusion preserves the energy score for training-like confident predictions, but increasingly replaces it with pooled-feature Mahalanobis distance when the classifier's MSP uncertainty is high. This targets OOS examples whose logits are ambiguous while retaining logit evidence for confident cases."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_uncertainty = np.asarray(ctx.h.msp(tr.logits), dtype=float)
    eval_uncertainty = np.asarray(ctx.h.msp(ev.logits), dtype=float)

    train_distance = np.asarray(gda.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_distance = np.asarray(gda.min_mahalanobis_sq(ev.pooled), dtype=float)

    e_lo, e_hi = np.quantile(train_energy, [0.05, 0.95])
    d_lo, d_hi = np.quantile(train_distance, [0.05, 0.95])
    u_lo, u_hi = np.quantile(train_uncertainty, [0.35, 0.75])

    e_scale = max(float(e_hi - e_lo), 1e-6)
    d_scale = max(float(d_hi - d_lo), 1e-6)
    u_scale = max(float(u_hi - u_lo), 1e-6)

    energy_score = np.clip((eval_energy - e_lo) / e_scale, -20.0, 20.0)
    distance_score = np.clip((eval_distance - d_lo) / d_scale, -20.0, 20.0)

    gate_input = np.clip((eval_uncertainty - u_lo) / u_scale, -20.0, 20.0)
    gate = 1.0 / (1.0 + np.exp(-6.0 * (gate_input - 0.5)))

    return (1.0 - gate) * energy_score + gate * distance_score
