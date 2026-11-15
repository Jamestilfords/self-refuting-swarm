import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Known-intent queries should be directionally aligned with the prototype of the intent predicted by the classifier. Class-conditional train calibration makes low predicted-prototype cosine comparable across intents while flagging confident assignments that lack representation support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0016", "m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)

    prototypes = np.zeros((k, x_train.shape[1]), dtype=float)
    global_proto = ctx.h.l2(x_train.mean(axis=0, keepdims=True))[0]

    for c in range(k):
        members = x_train[labels == c]
        if len(members):
            prototypes[c] = ctx.h.l2(members.mean(axis=0, keepdims=True))[0]
        else:
            prototypes[c] = global_proto

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_cos = np.sum(x_train * prototypes[train_pred], axis=1)
    eval_cos = np.sum(x_eval * prototypes[eval_pred], axis=1)

    centers = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)
    global_center = np.median(train_cos)
    global_scale = np.median(np.abs(train_cos - global_center)) + 1e-6

    for c in range(k):
        values = train_cos[train_pred == c]
        if len(values) >= 5:
            center = np.median(values)
            scale = np.median(np.abs(values - center))
            centers[c] = center
            scales[c] = max(scale, 1e-6)
        else:
            centers[c] = global_center
            scales[c] = global_scale

    scores = (centers[eval_pred] - eval_cos) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
