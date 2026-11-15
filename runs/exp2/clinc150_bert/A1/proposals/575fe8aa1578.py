import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Trailing-subspace distance detects atypical low-variance feature residuals, while MSP captures classifier uncertainty. Robust train-fitted scaling makes these complementary signals comparable without using evaluation-set statistics."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0007"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_dist = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_dist = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    train_msp = np.asarray(ctx.h.msp(tr.logits), dtype=float).reshape(-1)
    eval_msp = np.asarray(ctx.h.msp(ev.logits), dtype=float).reshape(-1)

    def robust_scale(train_values, values):
        center = np.median(train_values)
        q25, q75 = np.quantile(train_values, [0.25, 0.75])
        scale = (q75 - q25) / 1.349
        if not np.isfinite(scale) or scale < 1e-8:
            scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return np.clip((values - center) / scale, -20.0, 20.0)

    dist_z = robust_scale(train_dist, eval_dist)
    msp_z = robust_scale(train_msp, eval_msp)
    return dist_z + msp_z
