import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "Rather than using raw residual magnitude, this score measures the fraction of centered "
    "feature energy outside a global in-scope principal subspace and calibrates it per routed intent. "
    "OOS queries can have atypically large off-subspace semantic content even when their overall feature norm is ordinary."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    xtr = np.asarray(tr.pooled, dtype=np.float64)
    xev = np.asarray(ev.pooled, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)

    mean = np.mean(xtr, axis=0)
    ctr = xtr - mean
    cev = xev - mean

    cov = (ctr.T @ ctr) / max(len(ctr) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)

    rank = min(192, vecs.shape[1])
    basis = vecs[:, -rank:]

    def residual_fraction(x):
        total = np.sum(x * x, axis=1)
        projected = x @ basis
        explained = np.sum(projected * projected, axis=1)
        residual = np.maximum(total - explained, 0.0)
        return residual / np.maximum(total, 1e-12)

    tr_frac = residual_fraction(ctr)
    ev_frac = residual_fraction(cev)

    global_med = np.median(tr_frac)
    global_mad = np.median(np.abs(tr_frac - global_med))
    global_scale = max(1.4826 * global_mad, np.std(tr_frac), 1e-8)

    n_classes = int(max(
        np.max(labels) + 1 if labels.size else 1,
        tr.logits.shape[1],
        ev.logits.shape[1],
    ))

    medians = np.full(n_classes, global_med, dtype=np.float64)
    scales = np.full(n_classes, global_scale, dtype=np.float64)

    for c in range(n_classes):
        values = tr_frac[labels == c]
        if values.size >= 4:
            med = np.median(values)
            mad = np.median(np.abs(values - med))
            scale = max(1.4826 * mad, np.std(values), 1e-8)
            medians[c] = med
            scales[c] = scale

    routed = np.argmax(np.asarray(ev.logits), axis=1)
    routed = np.clip(routed, 0, n_classes - 1)

    score_values = (ev_frac - medians[routed]) / scales[routed]
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)
