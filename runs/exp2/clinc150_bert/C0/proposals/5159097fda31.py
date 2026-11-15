import numpy as np
from scipy.special import digamma

FAMILY = "other"
ARM = "edl"
RATIONALE = "OOS inputs can produce Dirichlet concentration shapes unlike those observed for any in-scope class, even when their total evidence or energy is high. This score measures class-conditional atypicality in predictive entropy and Dirichlet mutual information."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def shape_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        total = np.sum(alpha, axis=1, keepdims=True)
        mean = alpha / total
        entropy = -np.sum(mean * np.log(mean + eps), axis=1)
        expected_entropy = (
            digamma(total[:, 0] + 1.0)
            - np.sum(mean * digamma(alpha + 1.0), axis=1)
        )
        mutual_info = np.maximum(entropy - expected_entropy, 0.0)
        return np.column_stack((entropy, np.log(mutual_info + 1e-6)))

    z_train = shape_features(tr.logits)
    z_eval = shape_features(ev.logits)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    global_mean = np.mean(z_train, axis=0)
    global_var = np.var(z_train, axis=0) + 1e-5
    scores_by_class = []

    for c in classes:
        zc = z_train[labels == c]
        if zc.shape[0] < 2:
            mean = global_mean
            var = global_var
        else:
            mean = np.mean(zc, axis=0)
            var = np.var(zc, axis=0)
            var = np.maximum(var, 0.05 * global_var + 1e-5)
        scores_by_class.append(np.sum((z_eval - mean) ** 2 / var, axis=1))

    scores = np.min(np.column_stack(scores_by_class), axis=1)
    return scores
