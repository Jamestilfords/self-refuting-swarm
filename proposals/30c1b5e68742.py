import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The sorted cumulative Dirichlet-evidence profile measures how evidence is distributed "
    "across intents independently of total concentration and vacuity. Class-conditional "
    "deviation from the train-fitted profile flags atypically diffuse or concentrated evidence shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0269", "m_0204"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    k = a_tr.shape[1]

    ranks = np.unique(np.minimum(
        k, np.array([1, 2, 3, 5, 8, 12, 20, 32, 48, 64, 96, 128, k])
    )).astype(int)

    def profile(alpha):
        p = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), eps)
        ordered = np.sort(p, axis=1)[:, ::-1]
        cumulative = np.cumsum(ordered, axis=1)
        return cumulative[:, ranks - 1]

    x_tr = profile(a_tr)
    x_ev = profile(a_ev)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = x_tr.mean(axis=0)
    global_var = x_tr.var(axis=0)
    floor = np.maximum(global_var * 0.05, 1e-6)

    means = np.empty((k, x_tr.shape[1]), dtype=float)
    vars_ = np.empty((k, x_tr.shape[1]), dtype=float)

    for c in range(k):
        xc = x_tr[pred_tr == c]
        if xc.shape[0] >= 5:
            means[c] = xc.mean(axis=0)
            vars_[c] = xc.var(axis=0)
        else:
            means[c] = global_mean
            vars_[c] = global_var

    z = (x_ev - means[pred_ev]) / np.sqrt(np.maximum(vars_[pred_ev], floor))
    return np.mean(z * z, axis=1)
