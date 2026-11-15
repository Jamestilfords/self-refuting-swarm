import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Energy captures classifier uncertainty while pooled-space neighborhood distance measures local semantic support. Train-only empirical-CDF calibration puts these complementary signals on a common upper-tail scale before combining them."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_energy = ctx.h.energy(tr.logits)
    eval_energy = ctx.h.energy(ev.logits)

    train_knn = ctx.h.knn(tr.pooled, tr.pooled, k=21, l2=True)
    eval_knn = ctx.h.knn(tr.pooled, ev.pooled, k=21, l2=True)

    def ecdf_reference(train_values, values):
        reference = np.sort(np.asarray(train_values, dtype=float))
        values = np.asarray(values, dtype=float)
        q = np.searchsorted(reference, values, side="right") / float(reference.size)
        return np.clip(q, 0.0, 1.0)

    energy_tail = ecdf_reference(train_energy, eval_energy)
    knn_tail = ecdf_reference(train_knn, eval_knn)

    return 0.5 * energy_tail + 0.5 * knn_tail
