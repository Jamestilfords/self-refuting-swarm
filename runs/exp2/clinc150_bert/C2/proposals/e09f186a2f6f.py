import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Class means define directions used to discriminate known intents; OOS inputs often retain "
    "substantial feature energy in the orthogonal residual subspace. Training residual norms calibrate this score."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels)

    center = np.mean(x, axis=0)
    classes = np.unique(y)
    means = np.vstack([np.mean(x[y == c], axis=0) for c in classes])
    mean_offsets = means - center

    if len(classes) > 1:
        _, singular, vt = np.linalg.svd(mean_offsets, full_matrices=False)
        max_rank = min(len(classes) - 1, 64, vt.shape[0])
        if max_rank > 0:
            threshold = singular[0] * 1e-10 if singular.size else 0.0
            rank = min(max_rank, max(1, int(np.sum(singular > threshold))))
            basis = vt[:rank].T
            tr_centered = x - center
            ev_centered = z - center
            tr_residual = tr_centered - (tr_centered @ basis) @ basis.T
            ev_residual = ev_centered - (ev_centered @ basis) @ basis.T
        else:
            tr_residual = x - center
            ev_residual = z - center
    else:
        tr_residual = x - center
        ev_residual = z - center

    train_norm = np.sqrt(np.sum(tr_residual * tr_residual, axis=1))
    eval_norm = np.sqrt(np.sum(ev_residual * ev_residual, axis=1))
    scale = max(float(np.median(train_norm)), float(np.mean(train_norm)) * 0.1, 1e-8)
    return np.log1p(np.nan_to_num(eval_norm / scale, nan=0.0, posinf=1e6, neginf=0.0))
