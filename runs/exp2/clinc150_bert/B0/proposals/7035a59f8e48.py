import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries may be assigned a confident intent while having unusually weak cosine alignment "
    "with that intent's normalized feature prototype. Class-conditional calibration accounts for "
    "intents whose in-scope examples naturally have broader or tighter prototype alignment."
)
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0086"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = tr.pooled / (np.linalg.norm(tr.pooled, axis=1, keepdims=True) + eps)
    eval_x = ev.pooled / (np.linalg.norm(ev.pooled, axis=1, keepdims=True) + eps)

    n_classes = tr.logits.shape[1]
    sums = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(sums, tr.labels, train_x)

    global_proto = train_x.mean(axis=0)
    global_proto /= np.linalg.norm(global_proto) + eps

    prototypes = np.empty_like(sums)
    for c in range(n_classes):
        if counts[c] > 0:
            p = sums[c] / counts[c]
            prototypes[c] = p / (np.linalg.norm(p) + eps)
        else:
            prototypes[c] = global_proto

    train_pred = np.argmax(tr.logits, axis=1)
    train_cos = np.sum(train_x * prototypes[train_pred], axis=1)

    means = np.zeros(n_classes, dtype=float)
    scales = np.zeros(n_classes, dtype=float)
    global_mean = float(train_cos.mean())
    global_scale = float(np.std(train_cos))
    global_scale = max(global_scale, 1e-6)

    for c in range(n_classes):
        vals = train_cos[train_pred == c]
        if vals.size >= 2:
            means[c] = vals.mean()
            scales[c] = max(float(np.std(vals)), 1e-6)
        else:
            means[c] = global_mean
            scales[c] = global_scale

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(eval_x * prototypes[eval_pred], axis=1)

    return (means[eval_pred] - eval_cos) / scales[eval_pred]
