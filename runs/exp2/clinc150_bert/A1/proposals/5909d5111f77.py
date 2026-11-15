import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized pooled representations of in-scope queries should align with their nearest class prototype. "
    "Low maximum cosine similarity to normalized class prototypes indicates weak angular compatibility with known intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0037"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    n_classes = int(np.max(labels)) + 1
    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)

    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(proto_norm, 1e-12)

    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    x_eval = x_eval / np.maximum(eval_norm, 1e-12)

    max_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - max_cosine
