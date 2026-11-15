import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for high-confidence predictions, while adding a pooled-feature Mahalanobis penalty only when the classifier's maximum softmax probability is low. The gate and relative distance scale are calibrated entirely from in-scope training examples."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)
    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)

    d_ref = np.quantile(train_d, 0.75)
    d_mad = np.median(np.abs(train_d - np.median(train_d)))
    e_mad = np.median(np.abs(train_energy - np.median(train_energy)))
    d_scale = max(float(d_mad), 1e-6)
    e_scale = max(float(e_mad), 1e-6)

    confidence_cutoff = np.quantile(train_msp, 0.20)
    confidence_scale = max(
        float(np.quantile(train_msp, 0.75) - np.quantile(train_msp, 0.25)),
        1e-4,
    )
    gate = 1.0 / (1.0 + np.exp(np.clip(
        (eval_msp - confidence_cutoff) / confidence_scale, -40.0, 40.0
    )))

    excess_distance = np.maximum(eval_d - d_ref, 0.0) / d_scale
    score_values = eval_energy + gate * e_scale * excess_distance
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=-1e12)
