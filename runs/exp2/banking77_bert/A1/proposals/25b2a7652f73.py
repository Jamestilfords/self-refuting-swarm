import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "For logits that are typical for their predicted intent, the score preserves energy ordering. "
    "When class-conditional energy becomes uncertain, it smoothly switches toward pooled-space Mahalanobis novelty."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0105", "m_0109"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_lo, global_hi = np.quantile(e_tr, [0.05, 0.95])
    global_scale = max(global_hi - global_lo, 1e-8)

    lo = np.full(n_classes, global_lo, dtype=float)
    hi = np.full(n_classes, global_hi, dtype=float)
    gate_start = np.full(n_classes, np.quantile(e_tr, 0.70), dtype=float)
    gate_end = np.full(n_classes, np.quantile(e_tr, 0.92), dtype=float)

    for c in range(n_classes):
        values = e_tr[pred_tr == c]
        if values.size >= 8:
            q05, q70, q92, q95 = np.quantile(values, [0.05, 0.70, 0.92, 0.95])
            lo[c] = q05
            hi[c] = q95
            gate_start[c] = q70
            gate_end[c] = max(q92, q70 + 1e-8)

    e_norm = (e_ev - lo[pred_ev]) / np.maximum(hi[pred_ev] - lo[pred_ev], 1e-8)
    e_norm = np.clip(e_norm, -2.0, 3.0)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    d_lo, d_hi = np.quantile(d_tr, [0.05, 0.95])
    d_norm = (d_ev - d_lo) / max(d_hi - d_lo, 1e-8)
    d_norm = np.clip(d_norm, -2.0, 3.0)

    transition = np.maximum(gate_end[pred_ev] - gate_start[pred_ev], 1e-8)
    gate = np.clip((e_ev - gate_start[pred_ev]) / transition, 0.0, 1.0)

    return (1.0 - gate) * e_norm + gate * d_norm
