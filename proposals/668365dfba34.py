import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope examples should align with their classifier-selected prototype more strongly "
    "than with rival intent prototypes. A class-calibrated angular prototype margin flags queries "
    "whose representation is competitively ambiguous among known intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0020", "m_0202"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = ctx.h.l2(tr.pooled)
    xev = ctx.h.l2(ev.pooled)

    labels = tr.labels.astype(int)
    n_classes = tr.logits.shape[1]
    dim = xtr.shape[1]

    prototypes = np.zeros((n_classes, dim), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, xtr)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    train_sim = xtr @ prototypes.T
    train_own = train_sim[np.arange(xtr.shape[0]), labels]
    train_sim[np.arange(xtr.shape[0]), labels] = -np.inf
    train_rival = np.max(train_sim, axis=1)
    train_margin = train_own - train_rival

    global_mean = float(np.mean(train_margin))
    global_std = float(np.std(train_margin))
    global_std = max(global_std, 1e-6)

    class_mean = np.full(n_classes, global_mean, dtype=float)
    class_std = np.full(n_classes, global_std, dtype=float)
    for c in range(n_classes):
        values = train_margin[labels == c]
        if values.size >= 2:
            class_mean[c] = np.mean(values)
            class_std[c] = max(float(np.std(values)), 1e-6)

    pred = np.argmax(ev.logits, axis=1).astype(int)
    eval_sim = xev @ prototypes.T
    own = eval_sim[np.arange(xev.shape[0]), pred]
    eval_sim[np.arange(xev.shape[0]), pred] = -np.inf
    rival = np.max(eval_sim, axis=1)
    margin = own - rival

    return (class_mean[pred] - margin) / class_std[pred]
