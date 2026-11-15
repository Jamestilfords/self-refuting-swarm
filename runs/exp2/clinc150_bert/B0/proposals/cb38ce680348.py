import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope encoder representations should lie close to the dominant principal subspace of "
    "training features, while OOS inputs can retain unusually large residual components outside it."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean

    cov = centered_train.T @ centered_train
    cov /= max(centered_train.shape[0] - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    n_components = min(128, eigvecs.shape[1])
    principal = eigvecs[:, -n_components:]

    centered_eval = x_eval - mean
    projection = centered_eval @ principal
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    projected_sq = np.sum(projection * projection, axis=1)
    residual_sq = np.maximum(total_sq - projected_sq, 0.0)

    return np.nan_to_num(
        np.log1p(np.sqrt(residual_sq)),
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=0.0,
    )
