import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy ranking for high-MSP confident predictions, while progressively adding pooled-feature Mahalanobis distance when the classifier is uncertain."
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

    e_center = np.median(train_e)
    e_scale = np.quantile(train_e, 0.75) - np.quantile(train_e, 0.25)
    d_center = np.median(train_d)
    d_scale = np.quantile(train_d, 0.75) - np.quantile(train_d, 0.25)

    msp_cut = np.quantile(train_msp, 0.25)
    msp_scale = np.quantile(train_msp, 0.75) - np.quantile(train_msp, 0.25)

    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)
    msp_scale = max(float(msp_scale), 1e-6)

    uncertainty_gate = 1.0 / (1.0 + np.exp(
        np.clip((eval_msp - msp_cut) / msp_scale, -30.0, 30.0)
    ))

    energy_score = (eval_e - e_center) / e_scale
    distance_score = (eval_d - d_center) / d_scale
    return np.nan_to_num(energy_score + uncertainty_gate * distance_score,
                         nan=0.0, posinf=1e6, neginf=-1e6)
