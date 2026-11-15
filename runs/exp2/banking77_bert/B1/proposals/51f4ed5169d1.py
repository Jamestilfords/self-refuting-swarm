import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two margin and shift-invariant logit norm describe confidence shape beyond energy or maximum probability. A query whose confidence shape is atypical for its classifier-predicted intent receives a high class-conditional anomaly score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        k = logits.shape[1]
        top2 = np.partition(logits, k - 2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.mean(centered * centered, axis=1))
        return np.column_stack((margin, norm))

    x_train = shape_features(tr.logits)
    x_eval = shape_features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    counts = np.bincount(pred_train, minlength=n_classes).astype(float)
    sums = np.zeros((n_classes, 2), dtype=float)
    np.add.at(sums, pred_train, x_train)

    shrink = counts / (counts + 20.0)
    class_mean = sums / np.maximum(counts[:, None], 1.0)
    class_mean = shrink[:, None] * class_mean + (1.0 - shrink[:, None]) * global_mean

    delta = (x_eval - class_mean[pred_eval]) / np.sqrt(global_var)
    return np.sum(delta * delta, axis=1)
