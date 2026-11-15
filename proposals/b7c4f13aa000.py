import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy preserves the classifier's ordering for confident queries, while a train-calibrated "
    "feature-distance excess is activated only in the uncertain-energy regime. Conditioning the "
    "distance baseline on energy avoids treating naturally uncertain in-scope examples as OOS."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0012", "m_0026", "m_0021"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    # Standardize energy using only ID training statistics. This is an affine
    # transformation, so confident examples retain the energy ordering.
    e_med = np.median(e_tr)
    e_scale = np.median(np.abs(e_tr - e_med)) * 1.4826
    e_scale = max(float(e_scale), 1e-6)
    e_base = (e_ev - e_med) / e_scale

    # Estimate a robust distance reference separately across train-energy bands.
    # The residual is therefore "more distant than normal for this uncertainty".
    n_bins = 8
    edges = np.quantile(e_tr, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    tr_bin = np.clip(np.digitize(e_tr, edges[1:-1]), 0, n_bins - 1)
    ev_bin = np.clip(np.digitize(e_ev, edges[1:-1]), 0, n_bins - 1)

    d_center = np.empty(n_bins, dtype=float)
    d_scale = np.empty(n_bins, dtype=float)
    global_center = np.median(d_tr)
    global_scale = max(np.median(np.abs(d_tr - global_center)) * 1.4826, 1e-6)

    for b in range(n_bins):
        vals = d_tr[tr_bin == b]
        if vals.size < 8:
            d_center[b] = global_center
            d_scale[b] = global_scale
        else:
            center = np.median(vals)
            scale = np.median(np.abs(vals - center)) * 1.4826
            d_center[b] = center
            d_scale[b] = max(float(scale), 1e-6)

    d_excess = (d_ev - d_center[ev_bin]) / d_scale[ev_bin]
    d_excess = np.clip(d_excess, 0.0, 25.0)

    # Feature evidence is nearly absent below the train uncertainty shoulder and
    # smoothly becomes relevant above it.
    shoulder = float(np.quantile(e_tr, 0.60))
    gate_width = max(float(np.quantile(e_tr, 0.85) - shoulder), e_scale, 1e-6)
    gate = 1.0 / (1.0 + np.exp(-np.clip((e_ev - shoulder) / gate_width * 5.0,
                                        -40.0, 40.0)))

    return e_base + gate * np.log1p(d_excess)
