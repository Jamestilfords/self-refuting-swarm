import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The ranked cumulative Dirichlet-evidence curve measures how evidence mass is distributed "
    "across intents independently of total evidence. OOS inputs can exhibit an atypical evidence "
    "concentration spectrum for their predicted intent even when vacuity is uninformative."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def lorenz_features(alpha):
        alpha = np.nan_to_num(alpha, nan=1.0, posinf=1e6, neginf=1e-6)
        alpha = np.maximum(alpha, 1e-6)
        p = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        ranked = np.sort(p, axis=1)[:, ::-1]
        cumulative = np.cumsum(ranked, axis=1)

        k = ranked.shape[1]
        q = np.unique(np.linspace(1, k, min(16, k), dtype=int)) - 1
        curve = cumulative[:, q]

        # Logit-like coordinates emphasize both diffuse and winner-dominated shapes.
        curve = np.clip(curve, 1e-6, 1.0 - 1e-6)
        return np.log(curve / (1.0 - curve))

    train_alpha = ctx.h.alpha(tr.logits)
    eval_alpha = ctx.h.alpha(ev.logits)
    z_train = lorenz_features(train_alpha)
    z_eval = lorenz_features(eval_alpha)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    dim = z_train.shape[1]

    global_mean = z_train.mean(axis=0)
    global_var = z_train.var(axis=0) + 1e-5
    means = np.empty((n_classes, dim))
    variances = np.empty((n_classes, dim))

    for c in range(n_classes):
        members = z_train[train_pred == c]
        count = members.shape[0]
        if count < 4:
            means[c] = global_mean
            variances[c] = global_var
        else:
            mean_c = members.mean(axis=0)
            var_c = members.var(axis=0) + 1e-5
            shrink = min(20.0 / float(count), 0.8)
            means[c] = (1.0 - shrink) * mean_c + shrink * global_mean
            variances[c] = (1.0 - shrink) * var_c + shrink * global_var

    delta = z_eval - means[eval_pred]
    scores = np.sum((delta * delta) / variances[eval_pred], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
