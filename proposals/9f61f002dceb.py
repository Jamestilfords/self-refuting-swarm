import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should have both strong angular support from a prototype and a clear cosine "
    "margin over competing prototypes. Low train-calibrated support or prototype ambiguity indicates OOS."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020", "m_0075", "m_0299"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_tr = ctx.h.l2(tr.feats)
    x_ev = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = int(np.max(labels)) + 1

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = np.zeros((n_classes, x_tr.shape[1]), dtype=float)
    np.add.at(prototypes, labels, x_tr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    def support_and_margin(x):
        sims = x @ prototypes.T
        top2 = np.partition(sims, -2, axis=1)[:, -2:]
        support = top2[:, 1]
        margin = top2[:, 1] - top2[:, 0]
        return support, margin

    tr_support, tr_margin = support_and_margin(x_tr)
    ev_support, ev_margin = support_and_margin(x_ev)

    support_ref = np.sort(tr_support)
    margin_ref = np.sort(tr_margin)
    n = float(len(tr_support))

    support_cdf = np.searchsorted(support_ref, ev_support, side="right") / n
    margin_cdf = np.searchsorted(margin_ref, ev_margin, side="right") / n

    return -np.log(np.maximum(support_cdf, 1.0 / n)) - np.log(
        np.maximum(margin_cdf, 1.0 / n)
    )
