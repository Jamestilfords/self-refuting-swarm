import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "L2-normalized encoder features from an in-scope intent should align with at least one normalized class prototype, whereas OOS inputs tend to have lower maximum prototype cosine similarity."
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = []
MEMORY_USED = ["m_0131", "m_0132"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = tr.pooled / np.maximum(np.linalg.norm(tr.pooled, axis=1, keepdims=True), eps)
    eval_x = ev.pooled / np.maximum(np.linalg.norm(ev.pooled, axis=1, keepdims=True), eps)

    classes, inverse = np.unique(tr.labels, return_inverse=True)
    prototypes = np.zeros((len(classes), train_x.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, train_x)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), eps)

    max_cosine = np.max(eval_x @ prototypes.T, axis=1)
    return 1.0 - max_cosine
