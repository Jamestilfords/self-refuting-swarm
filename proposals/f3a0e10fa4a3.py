import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A supported classifier prediction should have stronger cosine alignment to its selected intent "
    "prototype than to competing intent prototypes. Class-conditional lower-tail calibration of this "
    "angular prototype margin flags unfamiliar or ambiguous intent geometry."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0122", "m_0127", "m_0126"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.feats)
    xev = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, xtr.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, xtr)
    global_proto = np.mean(xtr, axis=0, keepdims=True)
    global_proto = ctx.h.l2(global_proto)[0]
    valid = counts > 0
    prototypes[valid] /= counts[valid, None]
    prototypes[~valid] = global_proto
    prototypes = ctx.h.l2(prototypes)

    def angular_margin(x, logits):
        sim = x @ prototypes.T
        pred = np.argmax(logits, axis=1)
        chosen = sim[np.arange(sim.shape[0]), pred]
        masked = sim.copy()
        masked[np.arange(sim.shape[0]), pred] = -np.inf
        rival = np.max(masked, axis=1)
        return chosen - rival, pred

    train_margin, train_pred = angular_margin(xtr, tr.logits)
    eval_margin, eval_pred = angular_margin(xev, ev.logits)

    out = np.empty(eval_margin.shape[0], dtype=float)
    for c in range(n_classes):
        idx = np.flatnonzero(train_pred == c)
        test_idx = np.flatnonzero(eval_pred == c)
        if test_idx.size == 0:
            continue
        reference = train_margin[idx]
        if reference.size == 0:
            reference = train_margin
        reference = np.sort(reference)
        rank = np.searchsorted(reference, eval_margin[test_idx], side="left")
        lower_tail = (reference.size - rank) / float(reference.size)
        out[test_idx] = lower_tail - 1e-4 * eval_margin[test_idx]

    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
