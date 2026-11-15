import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "A smooth confidence gate preserves the energy score for high-MSP queries, while low-confidence queries increasingly use pooled-space Mahalanobis support. Train-calibrated standardization makes the two signals comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0074"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gda = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    e_train = ctx.h.energy(tr.logits)
    e_eval = ctx.h.energy(ev.logits)
    d_train = gda.min_mahalanobis_sq(tr.pooled)
    d_eval = gda.min_mahalanobis_sq(ev.pooled)

    e_center = np.median(e_train)
    d_center = np.median(d_train)
    e_scale = np.median(np.abs(e_train - e_center)) * 1.4826 + 1e-6
    d_scale = np.median(np.abs(d_train - d_center)) * 1.4826 + 1e-6

    e_z = (e_eval - e_center) / e_scale
    d_z = (d_eval - d_center) / d_scale

    msp_train = np.max(ctx.h.softmax(tr.logits), axis=1)
    msp_eval = np.max(ctx.h.softmax(ev.logits), axis=1)

    uncertain_cutoff = np.quantile(msp_train, 0.25)
    gate_scale = np.quantile(msp_train, 0.75) - np.quantile(msp_train, 0.25)
    gate_scale = max(gate_scale, 1e-4)

    gate_arg = np.clip((uncertain_cutoff - msp_eval) / gate_scale * 6.0, -40.0, 40.0)
    gate = 1.0 / (1.0 + np.exp(-gate_arg))

    return e_z + gate * (d_z - e_z)
