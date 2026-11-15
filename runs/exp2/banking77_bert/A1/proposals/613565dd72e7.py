import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Cosine agreement with a class prototype should be high for in-scope items, but raw agreement is not comparable across intents with different directional spread. A train-fitted class-conditional robust calibration makes low prototype alignment unusually suspicious relative to the classifier-selected intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0103"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    sums = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(sums, labels, x_train)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = np.sum(x_train * prototypes[labels], axis=1)
    global_med = np.median(train_cos)
    global_iqr = np.percentile(train_cos, 75) - np.percentile(train_cos, 25)
    global_scale = max(float(global_iqr), 1e-6)

    medians = np.full(n_classes, global_med, dtype=float)
    scales = np.full(n_classes, global_scale, dtype=float)
    for c in range(n_classes):
        values = train_cos[labels == c]
        if values.size >= 4:
            medians[c] = np.median(values)
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            scales[c] = max(float(iqr), 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[predicted], axis=1)
    standardized_surprise = (medians[predicted] - eval_cos) / scales[predicted]

    return np.tanh(standardized_surprise)
