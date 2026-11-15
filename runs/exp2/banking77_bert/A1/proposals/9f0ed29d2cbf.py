import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "In-scope representations tend to be substantially closer to one class prototype than to "
    "the runner-up prototype, while OOS representations often have similarly weak support from multiple intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    np.add.at(means, labels, x_train)

    global_mean = x_train.mean(axis=0)
    present = counts > 0
    means[present] /= counts[present, None]
    means[~present] = global_mean

    eval_norm = np.sum(x_eval * x_eval, axis=1, keepdims=True)
    mean_norm = np.sum(means * means, axis=1)[None, :]
    distances = eval_norm + mean_norm - 2.0 * (x_eval @ means.T)
    distances = np.maximum(distances, 0.0)

    nearest_two = np.partition(distances, 1, axis=1)[:, :2]
    d_near = np.minimum(nearest_two[:, 0], nearest_two[:, 1])
    d_second = np.maximum(nearest_two[:, 0], nearest_two[:, 1])

    return d_near / (d_second + 1e-12)
