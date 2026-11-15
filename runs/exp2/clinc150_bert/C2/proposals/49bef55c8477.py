import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck Mahalanobis density is sensitive to near-OOS queries that violate intent-level feature structure, while encoder-output density better captures broad domain shifts. Training-only standardization and precision weighting make their additive combination robust to differing score scales and redundancy."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = np.asarray(g_port.neg_log_density(tr.feats), dtype=float)
    full_train = np.asarray(g_full.neg_log_density(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.neg_log_density(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.neg_log_density(ev.pooled), dtype=float)

    mu_port = np.mean(port_train)
    mu_full = np.mean(full_train)
    sd_port = max(float(np.std(port_train)), 1e-8)
    sd_full = max(float(np.std(full_train)), 1e-8)

    z_train = np.column_stack((
        (port_train - mu_port) / sd_port,
        (full_train - mu_full) / sd_full,
    ))
    z_eval = np.column_stack((
        (port_eval - mu_port) / sd_port,
        (full_eval - mu_full) / sd_full,
    ))

    cov = np.cov(z_train, rowvar=False) + 1e-5 * np.eye(2)
    weights = np.linalg.solve(cov, np.ones(2))
    weights = np.maximum(weights, 1e-4)
    weights /= np.sum(weights)

    scores = z_eval @ weights
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
