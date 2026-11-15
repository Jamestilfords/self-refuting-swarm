import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query classified as an intent should be angularly closer to that intent's normalized prototype than to competing intent prototypes. A low predicted-prototype cosine margin indicates representation-logit disagreement and weak intent-specific support."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020", "m_0009", "m_0075", "m_0127"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = tr.labels
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_sim = x_train @ prototypes.T
    train_own = train_sim[np.arange(train_sim.shape[0]), labels]
    train_sim[np.arange(train_sim.shape[0]), labels] = -np.inf
    train_margin = train_own - np.max(train_sim, axis=1)

    class_med = np.zeros(n_classes, dtype=float)
    class_scale = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = train_margin[labels == c]
        if values.size:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            class_med[c] = med
            class_scale[c] = max(1.4826 * mad, 1e-4)

    pred = np.argmax(ev.logits, axis=1)
    eval_sim = x_eval @ prototypes.T
    own = eval_sim[np.arange(eval_sim.shape[0]), pred]
    eval_sim[np.arange(eval_sim.shape[0]), pred] = -np.inf
    margin = own - np.max(eval_sim, axis=1)

    return (class_med[pred] - margin) / class_scale[pred]
