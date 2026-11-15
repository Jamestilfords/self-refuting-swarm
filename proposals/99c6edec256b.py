FAMILY = "logit"
ARM = "ce"
RATIONALE = "The softmax-weighted variance and skewness of centered logits measure local thermal susceptibility of the winner-versus-rival distribution, rather than only its energy or maximum probability. Class-conditioned surprise identifies logit competition shapes that are unusual for the classifier-selected intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008"]


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def thermal_shape(logits):
        z = np.asarray(logits, dtype=float)
        z = z - np.max(z, axis=1, keepdims=True)
        p = np.exp(z)
        p /= np.sum(p, axis=1, keepdims=True)

        centered = z - np.sum(p * z, axis=1, keepdims=True)
        var = np.sum(p * centered * centered, axis=1)
        third = np.sum(p * centered * centered * centered, axis=1)

        log_var = np.log(var + eps)
        skew = third / np.power(var + eps, 1.5)
        return np.column_stack((log_var, skew))

    x_train = thermal_shape(tr.logits)
    x_eval = thermal_shape(ev.logits)

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    global_scale = np.std(x_train, axis=0) + 1e-4

    means = np.empty((k, 2), dtype=float)
    scales = np.empty((k, 2), dtype=float)

    for c in range(k):
        xc = x_train[pred_train == c]
        if xc.shape[0] >= 4:
            means[c] = np.mean(xc, axis=0)
            scales[c] = np.std(xc, axis=0) + 1e-4
        else:
            means[c] = global_mean
            scales[c] = global_scale

    residual = (x_eval - means[pred_eval]) / scales[pred_eval]
    return np.sum(residual * residual, axis=1)
