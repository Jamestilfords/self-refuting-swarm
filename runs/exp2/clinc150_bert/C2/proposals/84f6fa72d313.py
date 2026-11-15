import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This uses energy for high-confidence predictions while smoothly shifting toward full encoder-space Mahalanobis distance when classifier confidence is low, where logits are less reliable."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)

    e_med = np.median(train_e)
    d_med = np.median(train_d)
    e_scale = np.median(np.abs(train_e - e_med)) * 1.4826
    d_scale = np.median(np.abs(train_d - d_med)) * 1.4826
    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    confidence_center = np.quantile(train_msp, 0.50)
    confidence_scale = np.quantile(train_msp, 0.75) - np.quantile(train_msp, 0.25)
    confidence_scale = max(float(confidence_scale), 1e-4)

    uncertainty_gate = 1.0 / (
        1.0 + np.exp(np.clip((eval_msp - confidence_center) / confidence_scale, -40.0, 40.0))
    )

    energy_score = (eval_e - e_med) / e_scale
    distance_score = (eval_d - d_med) / d_scale
    scores = (1.0 - uncertainty_gate) * energy_score + uncertainty_gate * distance_score
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
