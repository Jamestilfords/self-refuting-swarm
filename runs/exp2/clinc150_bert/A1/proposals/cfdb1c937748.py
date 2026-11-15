import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A ridge map from pooled features to training logits identifies representation directions that "
    "the classifier actually uses to separate known intents. Large residual energy outside this "
    "logit-discriminative subspace indicates feature variation unsupported by in-scope decisions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0023"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y_train = np.asarray(tr.logits, dtype=np.float64)

    x_mean = np.mean(x_train, axis=0, keepdims=True)
    y_mean = np.mean(y_train, axis=0, keepdims=True)
    xc = x_train - x_mean
    yc = y_train - y_mean

    d = xc.shape[1]
    gram = xc.T @ xc
    ridge = 0.05 * (np.trace(gram) / max(d, 1) + 1e-8)
    coef = np.linalg.solve(gram + ridge * np.eye(d), xc.T @ yc)

    _, singular, vt = np.linalg.svd(coef.T, full_matrices=False)
    if singular.size:
        keep = singular > (singular[0] * 1e-7)
    else:
        keep = np.zeros(0, dtype=bool)

    if not np.any(keep):
        return np.sum((x_eval - x_mean) ** 2, axis=1)

    basis = vt[keep].T
    xe = x_eval - x_mean
    projected = xe @ basis
    residual = xe - projected @ basis.T
    return np.sum(residual * residual, axis=1)
