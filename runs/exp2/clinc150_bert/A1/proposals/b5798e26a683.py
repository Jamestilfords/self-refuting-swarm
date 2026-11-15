import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "Known queries should align with the prototype of the class predicted by the classifier. Standardizing predicted-class cosine alignment by that intent's training distribution flags atypical semantic directions while accounting for class-specific prototype spread."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    x_train = tr.feats / np.maximum(
        np.linalg.norm(tr.feats, axis=1, keepdims=True), eps
    )
    x_eval = ev.feats / np.maximum(
        np.linalg.norm(ev.feats, axis=1, keepdims=True), eps
    )

    prototypes = np.zeros((k, x_train.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(prototypes, tr.labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), eps
    )

    train_cos = np.sum(x_train * prototypes[tr.labels], axis=1)
    means = np.zeros(k, dtype=float)
    scales = np.ones(k, dtype=float)
    for c in range(k):
        values = train_cos[tr.labels == c]
        if values.size:
            means[c] = np.mean(values)
            scales[c] = max(np.std(values), 0.02)

    pred = np.argmax(ev.logits, axis=1)
    eval_cos = np.sum(x_eval * prototypes[pred], axis=1)
    z = (means[pred] - eval_cos) / scales[pred]
    return np.tanh(z / 3.0)
