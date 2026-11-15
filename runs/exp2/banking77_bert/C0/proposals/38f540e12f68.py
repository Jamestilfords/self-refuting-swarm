import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy remains the primary signal for confident predictions, while a train-calibrated uncertainty gate increases the contribution of pooled-feature Mahalanobis distance when classifier confidence is low. This targets inputs whose logits are ambiguous even when energy alone is not decisive."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    eval_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    train_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    eval_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    train_prob = ctx.h.softmax(tr.logits)
    eval_prob = ctx.h.softmax(ev.logits)
    train_msp = np.max(train_prob, axis=1)
    eval_msp = np.max(eval_prob, axis=1)

    def robust_scale(x):
        med = np.median(x)
        scale = np.median(np.abs(x - med)) * 1.4826
        return med, max(float(scale), 1e-6)

    e_med, e_scale = robust_scale(train_energy)
    d_med, d_scale = robust_scale(train_dist)

    msp_center = float(np.quantile(train_msp, 0.25))
    msp_scale = max(
        float(np.quantile(train_msp, 0.75) - np.quantile(train_msp, 0.25)),
        1e-4,
    )
    uncertainty_gate = 1.0 / (
        1.0 + np.exp(np.clip((eval_msp - msp_center) / msp_scale, -30.0, 30.0))
    )

    energy_z = (eval_energy - e_med) / e_scale
    distance_z = (eval_dist - d_med) / d_scale
    scores = energy_z + uncertainty_gate * distance_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
