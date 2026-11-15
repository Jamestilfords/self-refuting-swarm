import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy captures uncertainty in the classifier output, while normalized deep-nearest-neighbor "
    "distance measures whether an embedding has local support among in-scope examples. Train-fitted "
    "empirical tail maps make these heterogeneous signals comparable before requiring joint evidence."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = ["energy"]
MEMORY_USED = ["m_0003", "m_0055"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    # A compact PCA representation reduces noisy encoder directions before
    # measuring local normalized-neighbor support.
    pca = ctx.h.pca(tr.pooled, 192)
    z_tr = pca.transform(tr.pooled)
    z_ev = pca.transform(ev.pooled)

    energy_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    energy_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    # The self-neighbor occupies one rank for training examples, so use k+1
    # there to match the evaluation k-th non-self neighbor radius.
    knn_tr = np.asarray(ctx.h.knn(z_tr, z_tr, k=11, l2=True), dtype=float)
    knn_ev = np.asarray(ctx.h.knn(z_tr, z_ev, k=10, l2=True), dtype=float)

    def upper_tail_reference(train_values, values):
        ref = np.sort(np.asarray(train_values, dtype=float))
        n = ref.size
        # Smoothed upper-tail empirical probability, defined from train only.
        ranks = np.searchsorted(ref, values, side="right")
        return (ranks + 1.0) / (n + 2.0)

    e_tail = upper_tail_reference(energy_tr, energy_ev)
    k_tail = upper_tail_reference(knn_tr, knn_ev)

    eps = 1e-6
    e_tail = np.clip(e_tail, eps, 1.0 - eps)
    k_tail = np.clip(k_tail, eps, 1.0 - eps)

    # Add calibrated tail log-odds: a query is especially suspicious when both
    # classifier uncertainty and local representation unsupportedness are high.
    return np.log(e_tail / (1.0 - e_tail)) + np.log(k_tail / (1.0 - k_tail))
