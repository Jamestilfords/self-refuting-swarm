import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This preserves the energy score for training-like, confident predictions, while progressively emphasizing shared-covariance feature distance when classifier confidence is low. Ambiguous OOS queries can therefore be separated even when their energy alone resembles in-scope uncertainty."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)

    e_med = np.median(train_energy)
    e_scale = np.quantile(train_energy, 0.75) - np.quantile(train_energy, 0.25)
    d_med = np.median(train_d)
    d_scale = np.quantile(train_d, 0.75) - np.quantile(train_d, 0.25)

    e_scale = max(float(e_scale), 1e-6)
    d_scale = max(float(d_scale), 1e-6)

    confident_cut = np.quantile(train_msp, 0.50)
    msp_scale = np.quantile(train_msp, 0.75) - np.quantile(train_msp, 0.25)
    msp_scale = max(float(msp_scale), 1e-6)

    uncertainty_gate = 1.0 / (
        1.0 + np.exp(np.clip((eval_msp - confident_cut) / msp_scale, -40.0, 40.0))
    )

    energy_z = (eval_energy - e_med) / e_scale
    distance_z = (eval_d - d_med) / d_scale
    return energy_z + uncertainty_gate * distance_z
