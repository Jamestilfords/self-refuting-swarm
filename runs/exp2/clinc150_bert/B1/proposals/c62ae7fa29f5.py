import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "L2-normalized representations are compared with normalized class prototypes, while the "
    "entropy of prototype affinities captures whether a query has ambiguous angular support across intents."
)
SOURCE = "https://proceedings.mlr.press/v267/muller25a.html"
PARENTS = []
MEMORY_USED = ["m_0129", "m_0130"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    train_x = np.asarray(tr.pooled, dtype=float)
    eval_x = np.asarray(ev.pooled, dtype=float)
    labels = np.asarray(tr.labels)

    train_z = train_x / np.maximum(
        np.linalg.norm(train_x, axis=1, keepdims=True), eps
    )
    eval_z = eval_x / np.maximum(
        np.linalg.norm(eval_x, axis=1, keepdims=True), eps
    )

    classes, inverse = np.unique(labels, return_inverse=True)
    prototypes = np.zeros((classes.size, train_z.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, train_z)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), eps
    )

    train_cos = train_z @ prototypes.T
    own_cos = train_cos[np.arange(train_cos.shape[0]), inverse]
    scale = np.std(own_cos) + eps

    eval_cos = np.clip(eval_z @ prototypes.T, -1.0, 1.0)
    max_cos = np.max(eval_cos, axis=1)

    temperature = max(scale, 0.02)
    shifted = (eval_cos - np.max(eval_cos, axis=1, keepdims=True)) / temperature
    weights = np.exp(np.clip(shifted, -60.0, 0.0))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), eps)
    entropy = -np.sum(weights * np.log(np.maximum(weights, eps)), axis=1)
    entropy /= np.log(max(classes.size, 2))

    mismatch = np.maximum(0.0, np.mean(own_cos) - max_cos) / scale
    return mismatch * (0.5 + 0.5 * entropy)
