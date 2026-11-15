import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration profile captures how evidence is distributed among intents, "
    "separately from total evidence or vacuity. OOS queries can have an atypical class-conditional pattern "
    "of concentration entropy, winner dominance, and runner-up competition."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025", "m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def profile(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        p = alpha / np.sum(alpha, axis=1, keepdims=True)
        logp = np.log(p)

        entropy = -np.sum(p * logp, axis=1) / np.log(float(p.shape[1]))
        part = np.partition(p, -2, axis=1)
        second = part[:, -2]
        first = part[:, -1]

        dominance = first
        competition = second / np.maximum(first, eps)
        concentration = np.log(np.sum(alpha, axis=1))

        return np.column_stack((entropy, dominance, competition, concentration))

    x_train = profile(tr.logits)
    x_eval = profile(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-6

    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)

    for c in range(n_classes):
        xc = x_train[pred_train == c]
        if xc.shape[0] >= 4:
            means[c] = np.mean(xc, axis=0)
            variances[c] = np.var(xc, axis=0) + 0.05 * global_var + 1e-6
        else:
            means[c] = global_mean
            variances[c] = global_var

    residual = (x_eval - means[pred_eval]) / np.sqrt(variances[pred_eval])
    scores = np.sum(residual * residual, axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
