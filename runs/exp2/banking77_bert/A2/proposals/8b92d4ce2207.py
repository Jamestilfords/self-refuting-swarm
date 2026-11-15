import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope examples of each intent should exhibit a characteristic combination of total Dirichlet evidence, predictive concentration, and rival-intent mass. OOS inputs can be assigned a familiar class while having an atypical evidential shape for that class."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0085"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def evidential_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        total = np.sum(alpha, axis=1)
        prob = alpha / total[:, None]
        entropy = -np.sum(prob * np.log(prob + eps), axis=1)
        top = np.max(prob, axis=1)
        sq = np.sum(prob * prob, axis=1)
        return np.column_stack((
            np.log(total),
            entropy,
            top,
            sq,
        ))

    x_train = evidential_features(tr.logits)
    x_eval = evidential_features(ev.logits)
    labels = np.asarray(tr.labels, dtype=int)
    pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6
    means = np.empty((n_classes, dim))
    variances = np.empty((n_classes, dim))

    for c in range(n_classes):
        xc = x_train[labels == c]
        if xc.shape[0] >= 2:
            means[c] = np.mean(xc, axis=0)
            class_var = np.var(xc, axis=0)
            weight = xc.shape[0] / (xc.shape[0] + 10.0)
            variances[c] = weight * class_var + (1.0 - weight) * global_var
        elif xc.shape[0] == 1:
            means[c] = xc[0]
            variances[c] = global_var
        else:
            means[c] = global_mean
            variances[c] = global_var

    variances = np.maximum(variances, 1e-6)
    delta = x_eval - means[pred]
    score_value = np.sum(delta * delta / variances[pred], axis=1)
    score_value += np.sum(np.log(variances[pred]), axis=1)
    return np.nan_to_num(score_value, nan=0.0, posinf=1e12, neginf=-1e12)
