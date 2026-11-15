FAMILY = "knn"
ARM = "ce"
RATIONALE = "Jointly normalizing bottleneck and encoder features makes local support require agreement in both semantic representations. Averaging near and broader neighborhood radii reduces sensitivity to accidental close matches while retaining sensitivity to sparse OOS regions."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    tr_joint = np.concatenate(
        [ctx.h.l2(tr.feats), ctx.h.l2(tr.pooled)],
        axis=1,
    )
    ev_joint = np.concatenate(
        [ctx.h.l2(ev.feats), ctx.h.l2(ev.pooled)],
        axis=1,
    )

    n_train = tr_joint.shape[0]
    k_near = min(12, n_train)
    k_wide = min(64, n_train)

    d_near = ctx.h.knn(tr_joint, ev_joint, k=k_near, l2=True)
    d_wide = ctx.h.knn(tr_joint, ev_joint, k=k_wide, l2=True)
    out = 0.5 * (d_near + d_wide)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
