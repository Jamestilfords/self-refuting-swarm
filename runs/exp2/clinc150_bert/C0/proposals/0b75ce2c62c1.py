import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = "OOS inputs can induce atypical combinations of total Dirichlet concentration, predictive ambiguity, and concentration shape even when their energy overlaps with in-scope examples. A kNN distance in this evidential-shape space is therefore not a monotone transformation of energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    def evidential_shape(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-12)
        total = np.sum(alpha, axis=1)
        probs = alpha / total[:, None]

        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        entropy /= np.log(alpha.shape[1])

        concentration = np.sum(probs * probs, axis=1)

        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]

        return np.column_stack((
            np.log(total),
            entropy,
            concentration,
            margin,
        ))

    train_shape = evidential_shape(ctx.train.logits)
    eval_shape = evidential_shape(ctx.eval.logits)

    mean = np.mean(train_shape, axis=0)
    scale = np.std(train_shape, axis=0)
    scale = np.maximum(scale, 1e-6)

    train_shape = (train_shape - mean) / scale
    eval_shape = (eval_shape - mean) / scale

    scores = ctx.h.knn(train_shape, eval_shape, k=10, l2=False)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
