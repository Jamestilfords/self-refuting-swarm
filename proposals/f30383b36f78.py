import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "The norm outside a principal subspace fitted only on in-scope encoder features measures "
    "representation components unsupported by the known-intent manifold, as in ViM-style residual scoring."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0015", "m_0096"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.pooled, dtype=np.float64)
    z = np.asarray(ev.pooled, dtype=np.float64)

    mean = x.mean(axis=0)
    xc = x - mean

    # A fixed moderate rank retains shared in-scope semantic variation while
    # leaving unsupported encoder components in the residual space.
    rank = min(128, xc.shape[1], max(1, xc.shape[0] - 1))
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    basis = vt[:rank]

    residual = (z - mean) - ((z - mean) @ basis.T) @ basis
    scores = np.sqrt(np.sum(residual * residual, axis=1))

    # Avoid pathological exact constancy without using evaluation statistics.
    return scores + 1e-12 * np.sum((z - mean) * (z - mean), axis=1)
