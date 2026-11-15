import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A query confidently assigned to an intent can still be OOS if it lies unusually far from that "
    "predicted intent's representation prototype. Class-conditional robust radius normalization avoids "
    "treating naturally diffuse intents as equally suspicious as compact ones."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    np.add.at(sums, labels, x_train)
    means = sums / np.maximum(counts[:, None], 1.0)

    train_pred = np.argmax(tr.logits, axis=1)
    train_delta = x_train - means[train_pred]
    train_radius = np.sqrt(np.maximum(np.sum(train_delta * train_delta, axis=1), 0.0))

    medians = np.zeros(n_classes, dtype=np.float64)
    scales = np.zeros(n_classes, dtype=np.float64)
    global_median = np.median(train_radius)
    global_mad = np.median(np.abs(train_radius - global_median))
    global_scale = max(1.4826 * global_mad, 1e-6)

    for c in range(n_classes):
        values = train_radius[train_pred == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.05 * global_scale, 1e-6)
        else:
            medians[c] = global_median
            scales[c] = global_scale

    eval_pred = np.argmax(ev.logits, axis=1)
    eval_delta = x_eval - means[eval_pred]
    eval_radius = np.sqrt(np.maximum(np.sum(eval_delta * eval_delta, axis=1), 0.0))
    return (eval_radius - medians[eval_pred]) / scales[eval_pred]
