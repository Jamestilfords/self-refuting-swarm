import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope normalized representations should have broad cosine support from nearby intent "
    "prototypes, whereas OOS queries have low aggregate prototype affinity even when one prototype is accidentally close."
)
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0069", "m_0089"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)

    classes, inverse = np.unique(tr.labels, return_inverse=True)
    prototypes = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    cosine = x_eval @ prototypes.T
    temperature = 3.0
    shifted = temperature * cosine
    maximum = np.max(shifted, axis=1, keepdims=True)
    log_mean_affinity = (
        maximum[:, 0]
        + np.log(np.mean(np.exp(shifted - maximum), axis=1) + 1e-12)
    )
    return -log_mean_affinity
