import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and encoder-space densities capture complementary near-OOS and far-OOS deviations. "
    "Train-calibrated density percentiles make their tails comparable before an additive reliability-weighted fusion."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = np.asarray(g_port.neg_log_density(tr.feats), dtype=float)
    full_train = np.asarray(g_full.neg_log_density(tr.pooled), dtype=float)
    port_eval = np.asarray(g_port.neg_log_density(ev.feats), dtype=float)
    full_eval = np.asarray(g_full.neg_log_density(ev.pooled), dtype=float)

    port_train = np.nan_to_num(port_train, nan=0.0, posinf=1e12, neginf=-1e12)
    full_train = np.nan_to_num(full_train, nan=0.0, posinf=1e12, neginf=-1e12)
    port_eval = np.nan_to_num(port_eval, nan=0.0, posinf=1e12, neginf=-1e12)
    full_eval = np.nan_to_num(full_eval, nan=0.0, posinf=1e12, neginf=-1e12)

    port_sorted = np.sort(port_train)
    full_sorted = np.sort(full_train)

    port_pct = np.searchsorted(port_sorted, port_eval, side="right") / float(len(port_sorted))
    full_pct = np.searchsorted(full_sorted, full_eval, side="right") / float(len(full_sorted))

    port_iqr = np.percentile(port_train, 75) - np.percentile(port_train, 25)
    full_iqr = np.percentile(full_train, 75) - np.percentile(full_train, 25)

    port_weight = 1.0 / max(float(port_iqr), 1e-8)
    full_weight = 1.0 / max(float(full_iqr), 1e-8)
    total_weight = port_weight + full_weight

    scores = (port_weight * port_pct + full_weight * full_pct) / total_weight
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
