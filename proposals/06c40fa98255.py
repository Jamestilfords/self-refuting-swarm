import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents exhibit characteristic standardized rank-gap spectra: not only the winning "
    "margin, but how several competing logits fall away from the winner. Class-conditional "
    "deviation of this scale-invariant competition shape can flag OOS inputs with plausible "
    "confidence but atypical alternative-intent structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0033", "m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    def rank_shape(logits):
        z = logits - logits.mean(axis=1, keepdims=True)
        scale = np.sqrt(np.mean(z * z, axis=1, keepdims=True) + eps)
        s = np.sort(z, axis=1)[:, ::-1]
        n, k = s.shape
        width = min(5, k)
        gaps = (s[:, [0]] - s[:, 1:width]) / scale
        out = np.zeros((n, 4), dtype=float)
        if width > 1:
            out[:, :width - 1] = gaps
        return out

    x_train = rank_shape(tr.logits)
    x_eval = rank_shape(ev.logits)
    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    classes = tr.logits.shape[1]

    global_mean = x_train.mean(axis=0)
    global_var = x_train.var(axis=0) + 1e-5
    means = np.empty((classes, x_train.shape[1]), dtype=float)
    variances = np.empty_like(means)

    for c in range(classes):
        xc = x_train[pred_train == c]
        count = xc.shape[0]
        if count == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = xc.mean(axis=0)
            # Shrink unreliable predicted-intent variances toward global variation.
            local_var = xc.var(axis=0)
            weight = count / (count + 12.0)
            variances[c] = weight * local_var + (1.0 - weight) * global_var + 1e-6

    delta = x_eval - means[pred_eval]
    score_value = np.sum((delta * delta) / variances[pred_eval], axis=1)
    return np.log1p(np.maximum(score_value, 0.0))
