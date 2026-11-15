import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The normalized spectrum of ordered logit gaps captures the full winner-to-rival "
    "competition shape while discarding arbitrary logit offset and overall scale. "
    "Class-conditional deviation from the train-fitted gap-spectrum prototype flags atypical OOS competition patterns."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0190", "m_0255"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def gap_spectrum(logits):
        ordered = np.sort(logits, axis=1)[:, ::-1]
        gaps = ordered[:, :-1] - ordered[:, 1:]
        scale = np.sum(gaps, axis=1, keepdims=True)
        scale = np.maximum(scale, eps)
        return gaps / scale

    x_train = gap_spectrum(tr.logits)
    x_eval = gap_spectrum(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]
    d = x_train.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-5
    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)

    for c in range(k):
        members = x_train[train_pred == c]
        if members.shape[0] >= 3:
            means[c] = np.mean(members, axis=0)
            class_var = np.var(members, axis=0)
            shrink = members.shape[0] / (members.shape[0] + 20.0)
            variances[c] = shrink * class_var + (1.0 - shrink) * global_var + 1e-5
        else:
            means[c] = global_mean
            variances[c] = global_var

    delta = x_eval - means[eval_pred]
    return np.sum((delta * delta) / variances[eval_pred], axis=1)
