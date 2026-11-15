import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Rather than ranking only by residual magnitude, this score measures the fraction of a "
    "representation's centered energy lying outside the train-fitted principal subspace. "
    "It flags angular departures from the in-scope feature manifold while reducing sensitivity to overall feature norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0147", "m_0004"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = np.mean(x_train, axis=0)
    centered_train = x_train - mean

    _, singular_values, vt = np.linalg.svd(centered_train, full_matrices=False)
    variance = singular_values * singular_values
    cumulative = np.cumsum(variance)
    total = cumulative[-1] if cumulative.size else 1.0
    rank = int(np.searchsorted(cumulative, 0.90 * total) + 1)
    rank = max(1, min(rank, vt.shape[0] - 1))

    basis = vt[:rank]
    centered_eval = x_eval - mean
    total_sq = np.sum(centered_eval * centered_eval, axis=1)
    projected = centered_eval @ basis.T
    explained_sq = np.sum(projected * projected, axis=1)
    residual_sq = np.maximum(total_sq - explained_sq, 0.0)

    return residual_sq / (total_sq + 1e-12)
