import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope representation should both fit the classifier-predicted intent and be distinctly closer to that intent than to competing intent means. OOS queries tend to have a large predicted-class residual, a weak representation-space class margin, or both."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0073", "m_0083"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_own = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_own = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    train_other = train_d.copy()
    train_other[np.arange(train_other.shape[0]), train_pred] = np.inf
    eval_other = eval_d.copy()
    eval_other[np.arange(eval_other.shape[0]), eval_pred] = np.inf

    train_margin = np.min(train_other, axis=1) - train_own
    eval_margin = np.min(eval_other, axis=1) - eval_own

    own_center = np.median(train_own)
    margin_center = np.median(train_margin)
    own_scale = np.median(np.abs(train_own - own_center)) * 1.4826 + 1e-6
    margin_scale = np.median(np.abs(train_margin - margin_center)) * 1.4826 + 1e-6

    own_unusualness = (eval_own - own_center) / own_scale
    margin_ambiguity = (margin_center - eval_margin) / margin_scale
    return np.nan_to_num(own_unusualness + margin_ambiguity, nan=0.0, posinf=1e6, neginf=-1e6)
