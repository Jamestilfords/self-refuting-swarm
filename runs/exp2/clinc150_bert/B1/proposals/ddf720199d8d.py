import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Cosine alignment to the prototype of the classifier's predicted intent measures whether an "
    "example semantically supports its routed class. Training-fitted per-predicted-class calibration "
    "accounts for intents with naturally broader or tighter angular spread."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    def normalize(x):
        x = np.asarray(x, dtype=float)
        norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
        return x / np.maximum(norms, 1e-12)

    x_train = normalize(tr.pooled)
    x_eval = normalize(ev.pooled)

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = normalize(prototypes)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_cos = np.sum(x_train * prototypes[train_pred], axis=1)
    eval_cos = np.sum(x_eval * prototypes[eval_pred], axis=1)
    train_novelty = 1.0 - np.clip(train_cos, -1.0, 1.0)
    eval_novelty = 1.0 - np.clip(eval_cos, -1.0, 1.0)

    global_mean = float(np.mean(train_novelty))
    global_std = float(np.std(train_novelty))
    global_std = max(global_std, 1e-6)

    means = np.full(n_classes, global_mean, dtype=float)
    stds = np.full(n_classes, global_std, dtype=float)
    for cls in range(n_classes):
        values = train_novelty[train_pred == cls]
        if values.size >= 5:
            means[cls] = float(np.mean(values))
            stds[cls] = max(float(np.std(values)), 1e-6)

    return (eval_novelty - means[eval_pred]) / stds[eval_pred]
