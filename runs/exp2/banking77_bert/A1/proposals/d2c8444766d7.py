import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should have their classifier-predicted intent agree with the nearest "
    "class Gaussian prototype. OOS queries can be geometrically closer to another intent than "
    "the confidently predicted one, yielding an unusually large predicted-versus-alternative distance gap."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025", "m_0029"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.feats)
    eval_d = g.class_mahalanobis_sq(ev.feats)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    rows_train = np.arange(train_d.shape[0])
    rows_eval = np.arange(eval_d.shape[0])

    train_own = train_d[rows_train, train_pred]
    eval_own = eval_d[rows_eval, eval_pred]

    train_other = train_d.copy()
    eval_other = eval_d.copy()
    train_other[rows_train, train_pred] = np.inf
    eval_other[rows_eval, eval_pred] = np.inf

    train_gap = train_own - np.min(train_other, axis=1)
    eval_gap = eval_own - np.min(eval_other, axis=1)

    medians = np.zeros(train_d.shape[1], dtype=float)
    scales = np.ones(train_d.shape[1], dtype=float)
    global_med = np.median(train_gap)
    global_scale = np.median(np.abs(train_gap - global_med)) * 1.4826 + 1e-6

    for c in range(train_d.shape[1]):
        values = train_gap[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            scale = np.median(np.abs(values - med)) * 1.4826
            medians[c] = med
            scales[c] = max(scale, global_scale * 0.05, 1e-6)
        else:
            medians[c] = global_med
            scales[c] = global_scale

    return (eval_gap - medians[eval_pred]) / scales[eval_pred]
