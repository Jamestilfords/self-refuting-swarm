import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized bottleneck features remove representation magnitude and measure angular agreement with each intent prototype. OOS queries should have lower cosine similarity to every training class prototype than in-scope examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    train_norm = np.linalg.norm(x_train, axis=1, keepdims=True)
    train_norm = np.maximum(train_norm, 1e-12)
    x_train = x_train / train_norm

    eval_norm = np.linalg.norm(x_eval, axis=1, keepdims=True)
    eval_norm = np.maximum(eval_norm, 1e-12)
    x_eval = x_eval / eval_norm

    classes, inverse = np.unique(labels, return_inverse=True)
    prototypes = np.zeros((classes.size, x_train.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, x_train)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)

    proto_norm = np.linalg.norm(prototypes, axis=1, keepdims=True)
    prototypes /= np.maximum(proto_norm, 1e-12)

    best_cosine = np.max(x_eval @ prototypes.T, axis=1)
    return 1.0 - best_cosine
