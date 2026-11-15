import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Per-intent shrunk diagonal variances identify feature coordinates that are tightly constrained "
    "for a known intent, while residual-energy concentration catches atypical deviation patterns beyond total radius."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0027"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.feats, dtype=float)
    xev = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred_ev = np.argmax(np.asarray(ev.logits), axis=1)

    n, d = xtr.shape
    k = int(max(np.max(y), np.max(pred_ev))) + 1
    global_mean = np.mean(xtr, axis=0)
    global_var = np.var(xtr, axis=0) + 1e-6

    means = np.empty((k, d), dtype=float)
    vars_ = np.empty((k, d), dtype=float)
    radius_mu = np.empty(k, dtype=float)
    radius_sd = np.empty(k, dtype=float)
    entropy_mu = np.empty(k, dtype=float)
    entropy_sd = np.empty(k, dtype=float)

    for c in range(k):
        xc = xtr[y == c]
        if len(xc) == 0:
            xc = xtr
        nc = len(xc)
        mc = np.mean(xc, axis=0)
        vc = np.var(xc, axis=0)
        shrink = min(0.75, d / float(nc + d))
        vc = (1.0 - shrink) * vc + shrink * global_var + 1e-6

        means[c] = mc
        vars_[c] = vc

        q = (xc - mc) ** 2 / vc
        radius = np.mean(q, axis=1)
        weights = q / (np.sum(q, axis=1, keepdims=True) + 1e-12)
        concentration = np.sum(weights * np.log(weights * d + 1e-12), axis=1)

        radius_mu[c] = np.mean(radius)
        radius_sd[c] = np.std(radius) + 1e-6
        entropy_mu[c] = np.mean(concentration)
        entropy_sd[c] = np.std(concentration) + 1e-6

    mc = means[pred_ev]
    vc = vars_[pred_ev]
    qev = (xev - mc) ** 2 / vc
    radius_ev = np.mean(qev, axis=1)
    weights_ev = qev / (np.sum(qev, axis=1, keepdims=True) + 1e-12)
    concentration_ev = np.sum(weights_ev * np.log(weights_ev * d + 1e-12), axis=1)

    radius_z = (radius_ev - radius_mu[pred_ev]) / radius_sd[pred_ev]
    concentration_z = np.abs(
        (concentration_ev - entropy_mu[pred_ev]) / entropy_sd[pred_ev]
    )

    return radius_z + 0.35 * concentration_z
