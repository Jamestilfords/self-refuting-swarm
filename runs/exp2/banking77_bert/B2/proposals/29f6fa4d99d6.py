import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "L2-normalized in-scope features should align strongly with at least one normalized intent prototype, while OOS queries tend to have lower maximum cosine similarity to every class prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_x = np.asarray(tr.feats, dtype=float)
    eval_x = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    train_x = train_x / np.maximum(
        np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12
    )
    eval_norm = np.linalg.norm(eval_x, axis=1)
    eval_x = eval_x / np.maximum(eval_norm[:, None], 1e-12)

    n_classes = tr.logits.shape[1]
    prototypes = np.zeros((n_classes, train_x.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, train_x)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    present = counts > 0
    max_cosine = np.max(eval_x @ prototypes[present].T, axis=1)
    return 1.0 - max_cosine + 1e-12 * np.tanh(eval_norm)
