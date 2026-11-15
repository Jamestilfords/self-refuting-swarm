import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Class-conditional cosine alignment calibrates each predicted class by the in-scope distribution of training examples around its normalized prototype. OOS inputs that receive a class prediction but have unusually weak prototype alignment obtain high scores."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)

    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    train_cos = np.sum(x_train * prototypes[labels], axis=1)
    medians = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)

    for c in range(n_classes):
        vals = train_cos[labels == c]
        if vals.size:
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            medians[c] = med
            scales[c] = max(1.4826 * mad, 0.02)

    predicted = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[predicted], axis=1)
    scores = (medians[predicted] - eval_cos) / scales[predicted]
    return np.clip(scores, -20.0, 20.0)
