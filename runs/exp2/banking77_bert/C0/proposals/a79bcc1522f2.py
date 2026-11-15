import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "This keeps the energy ranking for confident predictions, while smoothly increasing the contribution of pooled-feature Mahalanobis distance when classifier uncertainty is high. Feature distance can therefore flag uncertain OOS items whose logits alone are ambiguous."
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_d = g.min_mahalanobis_sq(tr.pooled)
    eval_d = g.min_mahalanobis_sq(ev.pooled)

    train_e = ctx.h.energy(tr.logits)
    eval_e = ctx.h.energy(ev.logits)

    train_u = 1.0 - np.max(ctx.h.softmax(tr.logits), axis=1)
    eval_u = 1.0 - np.max(ctx.h.softmax(ev.logits), axis=1)

    def robust_scale(x):
        q25, q50, q75 = np.percentile(x, [25.0, 50.0, 75.0])
        return q50, max(q75 - q25, 1e-6)

    e_center, e_scale = robust_scale(train_e)
    d_center, d_scale = robust_scale(train_d)
    u_center, u_scale = robust_scale(train_u)

    z_energy = np.clip((eval_e - e_center) / e_scale, -50.0, 50.0)
    z_distance = np.clip((eval_d - d_center) / d_scale, -50.0, 50.0)

    uncertainty_gate = 1.0 / (
        1.0 + np.exp(-np.clip((eval_u - u_center) / u_scale, -30.0, 30.0))
    )
    return z_energy + uncertainty_gate * z_distance
