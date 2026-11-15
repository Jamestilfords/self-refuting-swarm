import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration profile encodes how evidence is distributed across "
    "intents independently of total evidence. OOS inputs should exhibit an atypical profile for "
    "their predicted intent even when vacuity or energy is not unusual."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0079", "m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def clr(alpha):
        a = np.clip(np.asarray(alpha, dtype=float), eps, 1e12)
        lp = np.log(a)
        return lp - np.mean(lp, axis=1, keepdims=True)

    z_tr = clr(ctx.h.alpha(tr.logits))
    z_ev = clr(ctx.h.alpha(ev.logits))
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]
    dim = z_tr.shape[1]

    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1.0)

    sums = np.zeros((n_classes, dim), dtype=float)
    sq_sums = np.zeros((n_classes, dim), dtype=float)
    np.add.at(sums, labels, z_tr)
    np.add.at(sq_sums, labels, z_tr * z_tr)

    means = sums / counts[:, None]
    variances = sq_sums / counts[:, None] - means * means
    global_var = np.var(z_tr, axis=0, keepdims=True)
    variances = np.maximum(variances, 0.05 * global_var + 1e-6)

    predicted = np.argmax(ev.logits, axis=1)
    residual = z_ev - means[predicted]
    return np.mean((residual * residual) / variances[predicted], axis=1)
