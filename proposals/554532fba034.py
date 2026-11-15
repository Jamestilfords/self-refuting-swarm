import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents have characteristic full winner-versus-rival logit competition profiles, not merely a typical maximum probability. "
    "A predicted-intent-conditioned diagonal surprise score over normalized top-rival gaps and logit scale detects atypical OOS competition shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0151", "m_0008", "m_0088", "m_0074"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = tr.logits.shape[1]
    n_rivals = min(5, max(1, k - 1))

    def features(logits):
        centered = logits - logits.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(centered * centered, axis=1) + eps)
        ordered = np.sort(logits, axis=1)[:, ::-1]
        if k > 1:
            gaps = (ordered[:, [0]] - ordered[:, 1:n_rivals + 1]) / scale[:, None]
        else:
            gaps = np.zeros((logits.shape[0], 1), dtype=float)
        return np.column_stack((np.log(scale + eps), gaps))

    x_train = features(tr.logits)
    x_eval = features(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-4
    means = np.empty((k, x_train.shape[1]), dtype=float)
    variances = np.empty_like(means)

    for cls in range(k):
        members = x_train[pred_train == cls]
        count = members.shape[0]
        if count == 0:
            means[cls] = global_mean
            variances[cls] = global_var
        else:
            weight = count / (count + 12.0)
            class_mean = members.mean(axis=0)
            class_var = members.var(axis=0)
            means[cls] = weight * class_mean + (1.0 - weight) * global_mean
            variances[cls] = (
                weight * class_var + (1.0 - weight) * global_var + 1e-4
            )

    residual = x_eval - means[pred_eval]
    score_values = np.sum(
        residual * residual / variances[pred_eval],
        axis=1,
    )
    return np.nan_to_num(score_values, nan=0.0, posinf=1e12, neginf=0.0)
