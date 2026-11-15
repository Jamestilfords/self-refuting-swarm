import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance within-class bottleneck directions encode constrained intent-specific structure, so OOS queries should have unusually large residual energy in this trailing subspace."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    model = ctx.h.partial(tr.feats, tr.labels, top_k=128)

    train_residual = np.asarray(model.bottom(tr.feats), dtype=float)
    eval_residual = np.asarray(model.bottom(ev.feats), dtype=float)

    finite_train = train_residual[np.isfinite(train_residual)]
    if finite_train.size == 0:
        finite_train = np.array([0.0, 1.0])
    reference = np.sort(finite_train)

    fill = reference[-1] + max(1.0, abs(reference[-1]))
    values = np.nan_to_num(eval_residual, nan=fill, posinf=fill, neginf=reference[0])
    return np.searchsorted(reference, values, side="right") / (reference.size + 1.0)
