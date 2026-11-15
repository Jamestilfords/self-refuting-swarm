import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A diagonal, class-specific feature model exposes dimensions that are unusually "
    "stable for each intent, while classifier-feature posterior disagreement captures "
    "queries whose logit evidence and class-conditional representation evidence conflict."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0036", "m_0040"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    y = np.asarray(tr.labels, dtype=np.int64)
    k = tr.logits.shape[1]

    # A fixed low-dimensional projection makes per-class diagonal scoring efficient
    # without constructing an item-by-class-by-feature tensor.
    rng = np.random.default_rng(271828)
    proj = rng.standard_normal((x_train.shape[1], 48))
    proj /= np.sqrt(np.sum(proj * proj, axis=0, keepdims=True) + 1e-12)
    z_train = x_train @ proj
    z_eval = x_eval @ proj

    global_mean = np.mean(z_train, axis=0)
    global_var = np.var(z_train, axis=0) + 1e-5
    means = np.empty((k, z_train.shape[1]), dtype=np.float64)
    variances = np.empty_like(means)
    priors = np.empty(k, dtype=np.float64)

    for c in range(k):
        mask = y == c
        n_c = int(np.sum(mask))
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
            priors[c] = 1.0 / k
        else:
            xc = z_train[mask]
            means[c] = np.mean(xc, axis=0)
            raw_var = np.var(xc, axis=0) + 1e-6
            shrink = n_c / (n_c + 20.0)
            variances[c] = shrink * raw_var + (1.0 - shrink) * global_var
            priors[c] = n_c / len(y)

    log_priors = np.log(np.maximum(priors, 1e-12))
    log_det = np.sum(np.log(variances), axis=1)

    def feature_log_post(z):
        out = np.empty((z.shape[0], k), dtype=np.float64)
        for c in range(k):
            diff = z - means[c]
            d2 = np.sum((diff * diff) / variances[c], axis=1)
            out[:, c] = -0.5 * (d2 + log_det[c]) + log_priors[c]
        out -= np.max(out, axis=1, keepdims=True)
        out -= np.log(np.sum(np.exp(out), axis=1, keepdims=True))
        return out

    def raw_components(z, logits):
        log_q = feature_log_post(z)
        p = ctx.h.softmax(logits)
        log_p = np.log(np.maximum(p, 1e-12))
        disagreement = np.sum(p * (log_p - log_q), axis=1)

        pred = np.argmax(logits, axis=1)
        diff = z - means[pred]
        pred_var = variances[pred]
        radial = np.sum((diff * diff) / pred_var, axis=1)
        return disagreement, radial

    train_dis, train_rad = raw_components(z_train, tr.logits)
    eval_dis, eval_rad = raw_components(z_eval, ev.logits)

    def robust_standardize(values, train_values):
        med = np.median(train_values)
        mad = np.median(np.abs(train_values - med))
        scale = max(1.4826 * mad, 1e-6)
        return np.clip((values - med) / scale, -20.0, 20.0)

    s_dis = robust_standardize(eval_dis, train_dis)
    s_rad = robust_standardize(eval_rad, train_rad)
    return s_dis + 0.65 * s_rad
