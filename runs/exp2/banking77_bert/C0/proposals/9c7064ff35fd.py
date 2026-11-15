import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "OOS queries may be assigned a confident classifier label while remaining unusually misaligned with that predicted class's normalized feature prototype. Class-conditional standardization accounts for differing within-class angular spread instead of ranking solely by the nearest prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]
    eps = 1e-12

    x_train = x_train / np.maximum(np.linalg.norm(x_train, axis=1, keepdims=True), eps)
    x_eval = x_eval / np.maximum(np.linalg.norm(x_eval, axis=1, keepdims=True), eps)

    global_proto = x_train.mean(axis=0)
    global_proto = global_proto / max(np.linalg.norm(global_proto), eps)

    prototypes = np.tile(global_proto, (n_classes, 1))
    means = np.zeros(n_classes, dtype=np.float64)
    scales = np.ones(n_classes, dtype=np.float64)

    for c in range(n_classes):
        members = x_train[labels == c]
        if members.shape[0] == 0:
            continue
        proto = members.mean(axis=0)
        proto = proto / max(np.linalg.norm(proto), eps)
        prototypes[c] = proto

        train_cos = members @ proto
        means[c] = np.mean(train_cos)
        scales[c] = max(np.std(train_cos), 0.02)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    cosine = np.sum(x_eval * prototypes[pred], axis=1)

    return (means[pred] - cosine) / scales[pred]
