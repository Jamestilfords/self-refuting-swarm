import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "For confident predictions this preserves the energy ranking, while high-entropy predictions "
    "progressively rely on distance from a probability-weighted mixture of intent prototypes. "
    "This makes uncertain queries suspicious when their representation is inconsistent with all intents suggested by the classifier."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0051", "m_0059"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    counts = np.bincount(y, minlength=k).astype(float)
    counts = np.maximum(counts, 1.0)
    means = np.zeros((k, xtr.shape[1]), dtype=float)
    np.add.at(means, y, xtr)
    means /= counts[:, None]

    true_resid = xtr - means[y]
    var = np.mean(true_resid * true_resid, axis=0)
    scale = np.median(var[var > 0]) if np.any(var > 0) else 1.0
    var = np.maximum(var, max(scale * 1e-5, 1e-8))

    ptr = ctx.h.softmax(tr.logits)
    pev = ctx.h.softmax(ev.logits)
    tr_mix = ptr @ means
    ev_mix = pev @ means

    dtr = np.sum((xtr - tr_mix) ** 2 / var, axis=1)
    dev = np.sum((xev - ev_mix) ** 2 / var, axis=1)

    etr = ctx.h.energy(tr.logits)
    eev = ctx.h.energy(ev.logits)
    htr = ctx.h.entropy(tr.logits)
    hev = ctx.h.entropy(ev.logits)

    def ecdf_fit(train_values, values):
        ordered = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ordered, values, side="right") / float(len(ordered) + 1)

    energy_rank = ecdf_fit(etr, eev)
    distance_rank = ecdf_fit(dtr, dev)

    low = np.quantile(htr, 0.65)
    high = np.quantile(htr, 0.95)
    gate = np.clip((hev - low) / (high - low + 1e-8), 0.0, 1.0)

    return (1.0 - gate) * energy_rank + gate * distance_rank
