import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "OOS queries can have weak angular support from the classifier-predicted intent "
    "and may instead align more closely with another intent prototype. A train-calibrated "
    "predicted-prototype shortfall plus prototype-disagreement margin captures both effects."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0020"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.pooled)
    x_eval = ctx.h.l2(ev.pooled)
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(norms, 1e-12)

    train_sim = x_train @ prototypes.T
    own_train_sim = train_sim[np.arange(labels.shape[0]), labels]

    class_mean = np.zeros(n_classes, dtype=float)
    class_std = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        values = own_train_sim[labels == c]
        if values.size:
            class_mean[c] = np.mean(values)
            class_std[c] = max(np.std(values), 0.02)

    eval_sim = x_eval @ prototypes.T
    predicted = np.argmax(ev.logits, axis=1)
    predicted_sim = eval_sim[np.arange(eval_sim.shape[0]), predicted]
    best_sim = np.max(eval_sim, axis=1)

    support_shortfall = (
        class_mean[predicted] - predicted_sim
    ) / class_std[predicted]
    prototype_disagreement = np.maximum(best_sim - predicted_sim, 0.0)

    return support_shortfall + 2.0 * prototype_disagreement
