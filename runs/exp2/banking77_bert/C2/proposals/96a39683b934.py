import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries may be assigned a class by the classifier but should have unusually weak cosine alignment with that predicted class's normalized feature prototype. Class-conditional standardization accounts for classes with naturally broader or tighter feature clusters."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    dim = x_train.shape[1]

    global_proto = x_train.mean(axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.tile(global_proto, (k, 1))
    means = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)

    valid = (labels >= 0) & (labels < k)
    for c in range(k):
        mask = valid & (labels == c)
        if np.any(mask):
            proto = x_train[mask].mean(axis=0)
            proto /= max(np.linalg.norm(proto), 1e-12)
            prototypes[c] = proto
            train_cos = x_train[mask] @ proto
            means[c] = np.mean(train_cos)
            scales[c] = max(np.std(train_cos), 0.02)

    predicted = np.argmax(ev.logits, axis=1)
    cosine = np.sum(x_eval * prototypes[predicted], axis=1)
    scores = (means[predicted] - cosine) / scales[predicted]
    return np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)
