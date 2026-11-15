import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS queries can deviate along low-variance within-class directions that are largely ignored by principal-subspace distances. The trailing-eigenspace score measures these atypical residual variations using a subspace fitted only on in-scope training features."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_scores = np.asarray(partial.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_scores = np.asarray(partial.bottom(ev.pooled), dtype=float).reshape(-1)

    finite_train = train_scores[np.isfinite(train_scores)]
    if finite_train.size == 0:
        finite_train = np.array([0.0, 1.0])

    center = np.median(finite_train)
    scale = np.median(np.abs(finite_train - center))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = np.std(finite_train)
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0

    eval_scores = np.nan_to_num(
        eval_scores,
        nan=center,
        posinf=center + 50.0 * scale,
        neginf=center - 50.0 * scale,
    )
    return np.clip((eval_scores - center) / scale, -50.0, 50.0)
