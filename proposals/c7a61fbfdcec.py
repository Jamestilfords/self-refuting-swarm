import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope queries should have both strong angular support from a class prototype and a clear "
    "angular margin over competing prototypes. Combining train-calibrated weak-support and low-margin "
    "tails detects OOS queries without reducing to nearest-prototype similarity alone."
)
SOURCE = "https://github.com/pradhyumnaag30/CLINC150"
PARENTS = []
MEMORY_USED = ["m_0020", "m_0009", "m_0075", "m_0122"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    n_classes = int(np.max(labels)) + 1

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_sim = x_train @ prototypes.T
    eval_sim = x_eval @ prototypes.T

    train_top2 = np.partition(train_sim, -2, axis=1)[:, -2:]
    eval_top2 = np.partition(eval_sim, -2, axis=1)[:, -2:]

    train_best = train_top2[:, 1]
    eval_best = eval_top2[:, 1]
    train_margin = train_top2[:, 1] - train_top2[:, 0]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]

    support_tail = np.searchsorted(
        np.sort(train_best), eval_best, side="right"
    ).astype(float) / float(len(train_best))

    margin_tail = np.searchsorted(
        np.sort(train_margin), eval_margin, side="right"
    ).astype(float) / float(len(train_margin))

    return 0.5 * ((1.0 - support_tail) + (1.0 - margin_tail))
