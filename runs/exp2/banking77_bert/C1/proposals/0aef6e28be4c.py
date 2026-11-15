import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "OOS inputs can receive a confident predicted intent while lying atypically far from that intent's class-conditional feature distribution. Calibrating predicted-class Mahalanobis distance against correctly classified training examples accounts for class-specific spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_d_all = g.class_mahalanobis_sq(tr.pooled)
    eval_d_all = g.class_mahalanobis_sq(ev.pooled)

    k = train_d_all.shape[1]
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_pred = np.clip(train_pred, 0, k - 1)
    eval_pred = np.clip(eval_pred, 0, k - 1)

    train_d = train_d_all[np.arange(train_d_all.shape[0]), train_pred]
    eval_d = eval_d_all[np.arange(eval_d_all.shape[0]), eval_pred]

    scores = np.empty(eval_d.shape[0], dtype=float)
    global_ref = np.sort(train_d[np.isfinite(train_d)])
    if global_ref.size == 0:
        global_ref = np.array([0.0])

    for c in range(k):
        mask_eval = eval_pred == c
        if not np.any(mask_eval):
            continue
        mask_train = (train_pred == c) & (tr.labels == c)
        ref = train_d[mask_train]
        ref = ref[np.isfinite(ref)]
        if ref.size < 8:
            ref = train_d[train_pred == c]
            ref = ref[np.isfinite(ref)]
        if ref.size < 2:
            ref = global_ref
        ref = np.sort(ref)
        ranks = np.searchsorted(ref, eval_d[mask_eval], side="right")
        tail = (ref.size - ranks + 1.0) / (ref.size + 1.0)
        scores[mask_eval] = -np.log(np.maximum(tail, 1e-12))

    return np.nan_to_num(scores, nan=0.0, posinf=30.0, neginf=0.0)
