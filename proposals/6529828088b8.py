import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = "A supported in-scope representation should remain locally dense as the neighborhood radius grows, whereas OOS points tend to have both a large near-neighbor distance and unusually rapid neighborhood expansion. Measuring this multiscale expansion after a train-fitted pooled-feature projection captures local support geometry beyond a single kNN radius."
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    pca = ctx.h.pca(tr.pooled, 192)
    train_x = pca.transform(tr.pooled)
    eval_x = pca.transform(ev.pooled)

    d_near = ctx.h.knn(train_x, eval_x, k=8, l2=True)
    d_far = ctx.h.knn(train_x, eval_x, k=64, l2=True)

    expansion = np.log((d_far + 1e-8) / (d_near + 1e-8))
    return d_near + 0.25 * expansion
