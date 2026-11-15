import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Each in-scope intent has a characteristic centered logit profile, including how strongly non-winning intents are suppressed. OOS queries that are assigned to a known intent but exhibit an atypical class-conditional logit shape receive a larger standardized residual score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    z_tr = np.asarray(tr.logits, dtype=float)
    z_ev = np.asarray(ev.logits, dtype=float)
    n, k = z_tr.shape

    x_tr = z_tr - np.mean(z_tr, axis=1, keepdims=True)
    x_ev = z_ev - np.mean(z_ev, axis=1, keepdims=True)

    labels = np.asarray(tr.labels, dtype=int)
    valid = (labels >= 0) & (labels < k)
    labels = labels[valid]
    x_fit = x_tr[valid]

    counts = np.bincount(labels, minlength=k).astype(float)
    sums = np.zeros((k, k), dtype=float)
    sums_sq = np.zeros((k, k), dtype=float)
    np.add.at(sums, labels, x_fit)
    np.add.at(sums_sq, labels, x_fit * x_fit)

    global_mean = np.mean(x_fit, axis=0)
    global_var = np.var(x_fit, axis=0) + 1e-6

    denom = np.maximum(counts[:, None], 1.0)
    class_mean = sums / denom
    class_mean[counts == 0] = global_mean

    raw_var = sums_sq / denom - class_mean * class_mean
    raw_var = np.maximum(raw_var, 0.0)

    shrink = 20.0
    class_var = (
        counts[:, None] * raw_var + shrink * global_var[None, :]
    ) / (counts[:, None] + shrink)
    class_var = np.maximum(class_var, 1e-6)

    pred = np.argmax(z_ev, axis=1)
    residual = x_ev - class_mean[pred]
    score_values = np.mean(
        (residual * residual) / class_var[pred],
        axis=1,
    )

    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)
