import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "An in-scope query should both lie near the classifier-predicted class and have that class geometrically separated from competing class supports. Large predicted-class Mahalanobis distance together with a small nearest-versus-second-nearest distance gap indicates representation-classifier inconsistency characteristic of OOS inputs."
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0089", "m_0090"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)
    g = ctx.h.gda(train_x, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(train_x)
    eval_d = g.class_mahalanobis_sq(eval_x)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_pred_d = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_pred_d = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    train_gap = train_two[:, 1] - train_two[:, 0]
    eval_gap = eval_two[:, 1] - eval_two[:, 0]

    d_med = np.median(train_pred_d)
    d_scale = np.median(np.abs(train_pred_d - d_med)) * 1.4826
    if not np.isfinite(d_scale) or d_scale < 1e-8:
        d_scale = np.std(train_pred_d) + 1e-8

    gap_med = np.median(train_gap)
    gap_scale = np.median(np.abs(train_gap - gap_med)) * 1.4826
    if not np.isfinite(gap_scale) or gap_scale < 1e-8:
        gap_scale = np.std(train_gap) + 1e-8

    score_out = (eval_pred_d - d_med) / d_scale - 0.5 * (eval_gap - gap_med) / gap_scale
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)
