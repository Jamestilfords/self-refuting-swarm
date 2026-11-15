import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The relative allocation of evidential mass among rival intents can be atypical for OOS inputs even when total evidence is high. "
    "A class-conditioned distance on the sorted non-predicted Dirichlet concentration spectrum isolates this evidence-shape deviation."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0061"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def rival_shape(logits, alpha):
        pred = np.argmax(logits, axis=1)
        a = np.nan_to_num(alpha, nan=eps, posinf=1e12, neginf=eps)
        a = np.maximum(a, eps)
        a = a.copy()
        a[np.arange(a.shape[0]), pred] = 0.0
        denom = np.maximum(a.sum(axis=1, keepdims=True), eps)
        rival = a / denom
        rival = np.sort(rival, axis=1)[:, ::-1]
        return np.log(np.maximum(rival[:, :-1], eps)), pred

    x_train, pred_train = rival_shape(tr.logits, ctx.h.alpha(tr.logits))
    x_eval, pred_eval = rival_shape(ev.logits, ctx.h.alpha(ev.logits))

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-4
    n_classes = tr.logits.shape[1]
    dim = x_train.shape[1]
    means = np.empty((n_classes, dim), dtype=float)
    variances = np.empty((n_classes, dim), dtype=float)
    prior = 20.0

    for c in range(n_classes):
        members = x_train[pred_train == c]
        count = members.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            weight = count / (count + prior)
            class_mean = members.mean(axis=0)
            class_var = members.var(axis=0)
            means[c] = weight * class_mean + (1.0 - weight) * global_mean
            variances[c] = (
                weight * class_var + (1.0 - weight) * global_var + 1e-4
            )

    delta = x_eval - means[pred_eval]
    out = np.mean((delta * delta) / variances[pred_eval], axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
