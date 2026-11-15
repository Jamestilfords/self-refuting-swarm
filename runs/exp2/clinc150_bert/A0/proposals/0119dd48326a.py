import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Directions that linearly explain class-logit variation are the classifier's discriminative "
    "feature subspace; OOS queries should retain unusually large energy in its orthogonal remainder."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=np.float64)
    z = np.asarray(tr.logits, dtype=np.float64)
    y = np.asarray(ev.feats, dtype=np.float64)

    mu_x = x.mean(axis=0)
    mu_z = z.mean(axis=0)
    xc = x - mu_x
    zc = z - mu_z

    scale = np.sqrt(np.mean(xc * xc, axis=0) + 1e-8)
    xs = xc / scale
    ys = (y - mu_x) / scale

    gram = xs.T @ xs
    ridge = 1e-3 * (np.trace(gram) / max(gram.shape[0], 1) + 1e-8)
    cross = xs.T @ zc
    weights = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), cross)

    _, singular, vt = np.linalg.svd(weights.T, full_matrices=False)
    keep = singular > (singular[0] * 1e-6 if singular.size else 0.0)
    basis = vt[keep].T
    if basis.shape[1] == 0:
        basis = np.eye(xs.shape[1], 1)

    q, _ = np.linalg.qr(basis, mode="reduced")
    residual = ys - (ys @ q) @ q.T
    return np.log1p(np.sum(residual * residual, axis=1))
