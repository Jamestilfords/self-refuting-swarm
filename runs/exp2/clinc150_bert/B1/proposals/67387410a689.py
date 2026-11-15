import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Temperature sensitivity of shift-invariant logits measures concentration shape rather than absolute energy: diffuse, weakly separated class evidence changes more under temperature scaling. Predicted-class train calibration removes natural intent-specific differences in logit sharpness."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0139"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def temperature_shape(logits, temperature=2.5):
        z = np.asarray(logits, dtype=float)
        z = z - np.max(z, axis=1, keepdims=True)

        a = z / temperature
        ma = np.max(a, axis=1, keepdims=True)
        logsum_t = (
            np.log(np.sum(np.exp(a - ma), axis=1)) + ma[:, 0]
        ) * temperature

        mb = np.max(z, axis=1, keepdims=True)
        logsum_1 = np.log(np.sum(np.exp(z - mb), axis=1)) + mb[:, 0]
        return logsum_t - logsum_1

    train_shape = temperature_shape(tr.logits)
    eval_shape = temperature_shape(ev.logits)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = float(np.mean(train_shape))
    global_std = float(np.std(train_shape))
    global_std = max(global_std, 1e-6)

    means = np.full(n_classes, global_mean, dtype=float)
    stds = np.full(n_classes, global_std, dtype=float)

    for c in range(n_classes):
        values = train_shape[train_pred == c]
        if values.size >= 8:
            means[c] = np.mean(values)
            stds[c] = max(float(np.std(values)), 1e-6)

    scores = (eval_shape - means[eval_pred]) / stds[eval_pred]
    return np.clip(scores, -50.0, 50.0)
