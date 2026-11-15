import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized class prototypes encode the angular support of known intents. A query is out-of-scope when even its nearest prototype has unusually weak cosine support relative to training examples assigned to that same prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0020"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    xtr = ctx.h.l2(tr.pooled)
    xev = ctx.h.l2(ev.pooled)

    sums = np.zeros((k, xtr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(sums, tr.labels, xtr)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    tr_sim = xtr @ prototypes.T
    tr_nearest = np.argmax(tr_sim, axis=1)
    tr_support = tr_sim[np.arange(tr_sim.shape[0]), tr_nearest]

    means = np.zeros(k, dtype=float)
    stds = np.ones(k, dtype=float)
    for c in range(k):
        values = tr_support[tr_nearest == c]
        if values.size:
            means[c] = values.mean()
            stds[c] = max(values.std(), 1e-6)

    ev_sim = xev @ prototypes.T
    ev_nearest = np.argmax(ev_sim, axis=1)
    ev_support = ev_sim[np.arange(ev_sim.shape[0]), ev_nearest]

    return (means[ev_nearest] - ev_support) / stds[ev_nearest]
