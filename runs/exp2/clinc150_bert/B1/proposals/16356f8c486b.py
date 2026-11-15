import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "L2-normalized pooled embeddings emphasize angular intent alignment rather than feature magnitude; OOS queries should have low cosine similarity to every in-scope class prototype."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=float)
    x_eval = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)

    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(proto_norm, 1e-12)

    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    similarities = x_eval @ prototypes.T
    valid = counts > 0
    similarities[:, ~valid] = -1.0
    return 1.0 - np.max(similarities, axis=1)
