import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should have high cosine alignment with the prototype of the intent "
    "predicted by the classifier. Class-conditional calibration flags unusually weak alignment "
    "without reducing to maximum prototype cosine."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, labels, x_train)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = np.sum(x_train * prototypes[labels], axis=1)
    cosine_sum = np.bincount(labels, weights=train_cos, minlength=n_classes)
    cosine_mean = cosine_sum / np.maximum(counts, 1.0)

    residual = train_cos - cosine_mean[labels]
    cosine_var = (
        np.bincount(labels, weights=residual * residual, minlength=n_classes)
        / np.maximum(counts, 1.0)
    )
    global_std = np.sqrt(np.mean(cosine_var) + 1e-8)
    cosine_std = np.sqrt(cosine_var + 0.1 * global_std * global_std + 1e-8)

    predicted = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[predicted], axis=1)
    score_out = (cosine_mean[predicted] - eval_cos) / cosine_std[predicted]

    return np.clip(score_out, -20.0, 20.0)
