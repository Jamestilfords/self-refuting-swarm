import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident queries retain their energy score exactly, while energy-uncertain queries "
    "are reranked by class-selected pooled-feature support calibrated among similarly uncertain training queries."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0169", "m_0250"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)
    d_tr = d_tr_all[np.arange(len(tr.pooled)), pred_tr]
    d_ev = d_ev_all[np.arange(len(ev.pooled)), pred_ev]

    k = tr.logits.shape[1]
    thresholds = np.empty(k, dtype=float)
    scales = np.empty(k, dtype=float)
    tail_refs = []

    global_threshold = np.quantile(e_tr, 0.70)
    global_scale = np.quantile(e_tr, 0.90) - global_threshold
    global_scale = max(float(global_scale), 1e-6)
    global_uncertain = d_tr[e_tr > global_threshold]
    if len(global_uncertain) < 8:
        global_uncertain = d_tr
    global_ref = np.sort(global_uncertain)

    for c in range(k):
        mask = pred_tr == c
        ec = e_tr[mask]
        dc = d_tr[mask]
        if len(ec) >= 12:
            threshold = float(np.quantile(ec, 0.70))
            scale = float(np.quantile(ec, 0.90) - threshold)
            uncertain = dc[ec > threshold]
            if len(uncertain) < 6:
                uncertain = dc
        else:
            threshold = float(global_threshold)
            scale = float(global_scale)
            uncertain = global_ref

        thresholds[c] = threshold
        scales[c] = max(scale, 1e-6)
        tail_refs.append(np.sort(uncertain))

    out = e_ev.copy()
    for c in range(k):
        mask = pred_ev == c
        if not np.any(mask):
            continue
        uncertain = e_ev[mask] > thresholds[c]
        if not np.any(uncertain):
            continue

        vals = d_ev[mask][uncertain]
        ref = tail_refs[c]
        tail = np.searchsorted(ref, vals, side="right") / (len(ref) + 1.0)
        feature_score = -np.log(np.maximum(1.0 - tail, 1e-8))
        routed = thresholds[c] + scales[c] * feature_score

        idx = np.flatnonzero(mask)[uncertain]
        out[idx] = routed

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
