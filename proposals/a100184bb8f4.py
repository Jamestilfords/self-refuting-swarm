import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The L4 norm of the encoder component outside a train-fitted principal subspace emphasizes "
    "sparse unsupported semantic deviations that can be diluted by the usual residual L2 norm."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0042", "m_0147"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)

    mean = x_train.mean(axis=0)
    centered_train = x_train - mean

    # A moderate rank retains dominant in-scope variation while leaving enough
    # dimensions to expose unsupported residual content.
    _, _, vt = np.linalg.svd(centered_train, full_matrices=False)
    rank = min(128, vt.shape[0])
    basis = vt[:rank]

    residual = (x_eval - mean) - ((x_eval - mean) @ basis.T) @ basis
    score_values = np.sum(residual ** 4, axis=1) ** 0.25
    return np.maximum(score_values, 0.0)
