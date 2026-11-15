import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query should be close to the mean of the intent predicted by the classifier, not merely close to some other known intent. Combining its predicted-class Mahalanobis distance with the disagreement between that distance and the nearest-class distance flags geometrically inconsistent predictions."
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = []
MEMORY_USED = ["m_0131", "m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    g = ctx.h.gda(x_train, tr.labels, cov="tied", whiten=True)
    d_train = np.asarray(g.class_mahalanobis_sq(x_train), dtype=float)
    d_eval = np.asarray(g.class_mahalanobis_sq(x_eval), dtype=float)

    p_train = np.argmax(tr.logits, axis=1)
    p_eval = np.argmax(ev.logits, axis=1)
    p_train = np.minimum(p_train, d_train.shape[1] - 1)
    p_eval = np.minimum(p_eval, d_eval.shape[1] - 1)

    row_train = np.arange(d_train.shape[0])
    row_eval = np.arange(d_eval.shape[0])
    own_train = d_train[row_train, p_train]
    own_eval = d_eval[row_eval, p_eval]
    gap_train = own_train - np.min(d_train, axis=1)
    gap_eval = own_eval - np.min(d_eval, axis=1)

    med_own = np.median(own_train)
    med_gap = np.median(gap_train)
    scale_own = np.median(np.abs(own_train - med_own)) + 1e-8
    scale_gap = np.median(np.abs(gap_train - med_gap)) + 1e-8

    out = (own_eval - med_own) / scale_own + (gap_eval - med_gap) / scale_gap
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
