import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and encoder densities capture complementary near- and far-OOS departures. "
    "Each density is converted to a train-calibrated survival score, then additive weights are fitted from held-out train density reliability."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n = len(tr.labels)

    # Deterministic held-out train split estimates which representation yields
    # more stable density values on unseen in-scope examples.
    rng = np.random.default_rng(271828)
    perm = rng.permutation(n)
    cut = max(1, int(0.75 * n))
    fit_idx = perm[:cut]
    val_idx = perm[cut:]

    g_port_hold = ctx.h.gda(
        tr.feats[fit_idx], tr.labels[fit_idx], cov="tied"
    )
    g_full_hold = ctx.h.gda(
        tr.pooled[fit_idx], tr.labels[fit_idx], cov="tied"
    )

    if len(val_idx) > 0:
        port_hold = g_port_hold.neg_log_density(tr.feats[val_idx])
        full_hold = g_full_hold.neg_log_density(tr.pooled[val_idx])
        port_rel = np.std(port_hold) + 1e-6
        full_rel = np.std(full_hold) + 1e-6
    else:
        port_rel = 1.0
        full_rel = 1.0

    # Favor the representation whose held-out in-scope density is more stable.
    reliability = np.array([1.0 / port_rel, 1.0 / full_rel])
    weights = reliability / np.sum(reliability)

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.neg_log_density(tr.feats)
    full_train = g_full.neg_log_density(tr.pooled)
    port_eval = g_port.neg_log_density(ev.feats)
    full_eval = g_full.neg_log_density(ev.pooled)

    # Smoothed empirical survival probabilities produce nonlinear tail evidence,
    # avoiding a near-duplicate of raw standardized density addition.
    port_sorted = np.sort(port_train)
    full_sorted = np.sort(full_train)

    port_rank = np.searchsorted(port_sorted, port_eval, side="right")
    full_rank = np.searchsorted(full_sorted, full_eval, side="right")

    port_survival = (n - port_rank + 1.0) / (n + 2.0)
    full_survival = (n - full_rank + 1.0) / (n + 2.0)

    port_tail = -np.log(np.clip(port_survival, 1e-12, 1.0))
    full_tail = -np.log(np.clip(full_survival, 1e-12, 1.0))

    return weights[0] * port_tail + weights[1] * full_tail
