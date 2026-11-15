import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Cosine compatibility with the prototype of the classifier-predicted intent is calibrated "
    "against that intent's training compatibility distribution, so unusually weak angular support is OOS-like."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0068"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    train_x = tr.feats / (np.linalg.norm(tr.feats, axis=1, keepdims=True) + eps)
    eval_x = ev.feats / (np.linalg.norm(ev.feats, axis=1, keepdims=True) + eps)

    prototypes = np.zeros((k, train_x.shape[1]), dtype=float)
    global_proto = train_x.mean(axis=0)
    global_proto /= np.linalg.norm(global_proto) + eps

    for c in range(k):
        members = train_x[tr.labels == c]
        if members.shape[0] > 0:
            proto = members.mean(axis=0)
            prototypes[c] = proto / (np.linalg.norm(proto) + eps)
        else:
            prototypes[c] = global_proto

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_support = np.sum(train_x * prototypes[train_pred], axis=1)
    eval_support = np.sum(eval_x * prototypes[eval_pred], axis=1)

    scores = np.empty(eval_support.shape[0], dtype=float)
    global_mean = float(np.mean(train_support))
    global_std = float(np.std(train_support))
    global_std = max(global_std, 1e-6)

    for c in range(k):
        mask_train = train_pred == c
        mask_eval = eval_pred == c

        if not np.any(mask_eval):
            continue

        class_support = train_support[mask_train]
        if class_support.shape[0] >= 4:
            mean = float(np.mean(class_support))
            std = max(float(np.std(class_support)), 1e-6)
            shrink = class_support.shape[0] / (class_support.shape[0] + 10.0)
            mean = shrink * mean + (1.0 - shrink) * global_mean
            std = shrink * std + (1.0 - shrink) * global_std
        else:
            mean = global_mean
            std = global_std

        scores[mask_eval] = (mean - eval_support[mask_eval]) / std

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
