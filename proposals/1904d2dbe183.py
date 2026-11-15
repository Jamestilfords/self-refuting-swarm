import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "An in-scope query should align angularly with the prototype of the intent selected by the classifier. "
    "Class-conditional calibration of that selected-prototype similarity flags atypically weak support without favoring broad intent clusters."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020", "m_0009", "m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    train_x = ctx.h.l2(tr.pooled)
    eval_x = ctx.h.l2(ev.pooled)

    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, train_x)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_sim = np.sum(train_x * prototypes[train_pred], axis=1)
    eval_sim = np.sum(eval_x * prototypes[eval_pred], axis=1)

    global_mean = float(np.mean(train_sim))
    global_std = float(np.std(train_sim))
    global_std = max(global_std, 1e-6)

    means = np.full(n_classes, global_mean, dtype=float)
    stds = np.full(n_classes, global_std, dtype=float)

    for c in range(n_classes):
        values = train_sim[train_pred == c]
        if values.size >= 4:
            # Mild shrinkage prevents unstable calibration for rarely predicted intents.
            weight = values.size / (values.size + 10.0)
            means[c] = weight * np.mean(values) + (1.0 - weight) * global_mean
            local_std = float(np.std(values))
            stds[c] = max(
                weight * local_std + (1.0 - weight) * global_std,
                1e-6,
            )

    return (means[eval_pred] - eval_sim) / stds[eval_pred]
