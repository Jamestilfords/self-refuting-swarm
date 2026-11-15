import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Angular distance to the classifier-predicted intent prototype tests whether an embedding points in the "
    "semantic direction typical of that intent. Per-predicted-class robust scaling accommodates intents with "
    "different directional dispersion."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = np.asarray(tr.pooled, dtype=float)
    eval_x = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    train_x = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), eps)
    eval_x = eval_x / np.maximum(np.linalg.norm(eval_x, axis=1, keepdims=True), eps)

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    centers = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    global_proto = np.mean(train_x, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), eps)

    for c in range(n_classes):
        members = train_x[labels == c]
        if members.shape[0] == 0:
            prototypes[c] = global_proto
            continue

        proto = np.mean(members, axis=0)
        proto /= max(np.linalg.norm(proto), eps)
        prototypes[c] = proto

        train_dist = 1.0 - np.clip(members @ proto, -1.0, 1.0)
        med = np.median(train_dist)
        mad = np.median(np.abs(train_dist - med))
        centers[c] = med
        scales[c] = max(1.4826 * mad, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    dist = 1.0 - np.clip(np.sum(eval_x * prototypes[pred], axis=1), -1.0, 1.0)
    scores = (dist - centers[pred]) / scales[pred]

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
