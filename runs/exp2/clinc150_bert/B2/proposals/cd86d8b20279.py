import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "Averaging several local neighbor distances is less sensitive than a single boundary neighbor while measuring whether a normalized encoder representation has sustained in-scope support. OOS queries should lie farther from a neighborhood of training examples even if one accidental neighbor is close."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    # Average several fixed-order local radii, using only training support.
    radii = []
    for k in (5, 10, 20, 40):
        radii.append(ctx.h.knn(x_train, x_eval, k=k, l2=False))
    score_out = np.mean(np.column_stack(radii), axis=1)

    # Avoid a degenerate constant vector while preserving ordering in normal cases.
    if np.all(score_out == score_out[0]):
        score_out = score_out + 1e-12 * np.arange(score_out.shape[0])
    return np.nan_to_num(score_out, nan=0.0, posinf=1e12, neginf=-1e12)
