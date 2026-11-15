import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should be unusually compatible with the representation distribution of "
    "the class predicted by the classifier, rather than merely close to some class. A small "
    "nearest-versus-second-nearest Mahalanobis gap additionally flags ambiguous feature assignments."
)
SOURCE = "https://arxiv.org/abs/2106.09022"
PARENTS = []
MEMORY_USED = ["m_0116", "m_0128"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d = g.class_mahalanobis_sq(tr.pooled)
    eval_d = g.class_mahalanobis_sq(ev.pooled)
    n_classes = train_d.shape[1]

    train_pred = np.clip(np.argmax(tr.logits, axis=1), 0, n_classes - 1)
    eval_pred = np.clip(np.argmax(ev.logits, axis=1), 0, n_classes - 1)

    train_self = train_d[np.arange(train_d.shape[0]), train_pred]
    eval_self = eval_d[np.arange(eval_d.shape[0]), eval_pred]

    train_two = np.partition(train_d, 1, axis=1)[:, :2]
    eval_two = np.partition(eval_d, 1, axis=1)[:, :2]
    train_gap = train_two[:, 1] - train_two[:, 0]
    eval_gap = eval_two[:, 1] - eval_two[:, 0]

    global_self_med = np.median(train_self)
    global_self_scale = np.quantile(train_self, 0.9) - global_self_med
    global_gap_med = np.median(train_gap)
    global_gap_scale = np.quantile(train_gap, 0.9) - global_gap_med

    self_med = np.full(n_classes, global_self_med, dtype=float)
    self_scale = np.full(n_classes, global_self_scale, dtype=float)
    gap_med = np.full(n_classes, global_gap_med, dtype=float)
    gap_scale = np.full(n_classes, global_gap_scale, dtype=float)

    for c in range(n_classes):
        mask = train_pred == c
        if np.sum(mask) >= 8:
            own_self = train_self[mask]
            own_gap = train_gap[mask]
            self_med[c] = np.median(own_self)
            self_scale[c] = np.quantile(own_self, 0.9) - self_med[c]
            gap_med[c] = np.median(own_gap)
            gap_scale[c] = np.quantile(own_gap, 0.9) - gap_med[c]

    self_scale = np.maximum(self_scale, 1e-6)
    gap_scale = np.maximum(gap_scale, 1e-6)

    self_tail = (eval_self - self_med[eval_pred]) / self_scale[eval_pred]
    ambiguity = (eval_gap - gap_med[eval_pred]) / gap_scale[eval_pred]
    scores = self_tail - 0.35 * ambiguity

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
