import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "OOS points tend to have both a large local neighbor radius and unusually rapid "
    "neighborhood expansion at a broader radius. Class-calibrating this multiscale "
    "pooled-space support score avoids treating naturally diffuse intents as OOS."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0123"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    # For training queries, the nearest neighbor is ordinarily the query itself,
    # so use k+1 to obtain radii comparable to evaluation-time k radii.
    tr_near = ctx.h.knn(tr.pooled, tr.pooled, k=9, l2=True)
    tr_far = ctx.h.knn(tr.pooled, tr.pooled, k=49, l2=True)
    ev_near = ctx.h.knn(tr.pooled, ev.pooled, k=8, l2=True)
    ev_far = ctx.h.knn(tr.pooled, ev.pooled, k=48, l2=True)

    eps = 1e-8
    # The first term is broad support; the second detects sparse local
    # neighborhoods whose radius grows unusually quickly with k.
    tr_raw = np.log1p(np.maximum(tr_far, 0.0)) + 0.55 * np.log(
        (np.maximum(tr_far, 0.0) + eps) / (np.maximum(tr_near, 0.0) + eps)
    )
    ev_raw = np.log1p(np.maximum(ev_far, 0.0)) + 0.55 * np.log(
        (np.maximum(ev_far, 0.0) + eps) / (np.maximum(ev_near, 0.0) + eps)
    )

    global_med = np.median(tr_raw)
    global_mad = np.median(np.abs(tr_raw - global_med))
    global_scale = max(1.4826 * global_mad, 1e-6)

    k = tr.logits.shape[1]
    med = np.full(k, global_med, dtype=float)
    scale = np.full(k, global_scale, dtype=float)

    for c in range(k):
        values = tr_raw[tr.labels == c]
        if values.size >= 8:
            center = np.median(values)
            spread = 1.4826 * np.median(np.abs(values - center))
            med[c] = center
            scale[c] = max(spread, global_scale * 0.25, 1e-6)

    pred = np.argmax(ev.logits, axis=1)
    out = (ev_raw - med[pred]) / scale[pred]

    # Deterministic negligible tie-breaking while preserving the multiscale kNN score.
    out = out + 1e-7 * ctx.h.energy(ev.logits)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
