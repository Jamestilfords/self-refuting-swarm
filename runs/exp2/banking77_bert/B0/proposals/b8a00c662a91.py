import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "The span of class means captures directions that discriminate known intents, while residual components describe variation not explained by any in-scope intent separation. OOS queries with atypical residual structure receive larger scores."
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0003"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels)
    center = x.mean(axis=0)

    classes = np.unique(y)
    means = np.vstack([x[y == c].mean(axis=0) for c in classes])
    between = means - center

    _, s, vt = np.linalg.svd(between, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        residual = ev.feats - center
    else:
        rank = max(1, int(np.sum(s > s[0] * 1e-8)))
        basis = vt[:rank]
        z = ev.feats - center
        residual = z - (z @ basis.T) @ basis

    return np.sum(residual * residual, axis=1)
