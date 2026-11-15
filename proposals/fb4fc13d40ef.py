import numpy as np
from sklearn.decomposition import PCA

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope bottleneck features should lie near the principal subspace learned from training intents. "
    "The centered residual norm measures semantic feature components unsupported by that in-scope manifold."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0004", "m_0147", "m_0194"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    dim = min(96, tr.feats.shape[1], max(1, tr.feats.shape[0] - 1))
    pca = PCA(n_components=dim, svd_solver="randomized", random_state=2718)
    pca.fit(tr.feats)

    tr_resid = tr.feats - pca.inverse_transform(pca.transform(tr.feats))
    ev_resid = ev.feats - pca.inverse_transform(pca.transform(ev.feats))

    tr_norm = np.sqrt(np.sum(tr_resid * tr_resid, axis=1))
    ev_norm = np.sqrt(np.sum(ev_resid * ev_resid, axis=1))

    center = np.median(tr_norm)
    scale = np.median(np.abs(tr_norm - center)) * 1.4826
    scale = max(float(scale), 1e-8)
    return np.clip((ev_norm - center) / scale, -50.0, 50.0)
