import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Confident classifier outputs retain the energy score, while energy-uncertain outputs are "
    "rescored by predicted-intent representation atypicality. Class-conditional calibration "
    "prevents naturally broad intents from receiving excessive feature-distance penalties."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    e_tr = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    e_ev = np.asarray(ctx.h.energy(ev.logits), dtype=float)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr_all = np.asarray(g.class_mahalanobis_sq(tr.pooled), dtype=float)
    d_ev_all = np.asarray(g.class_mahalanobis_sq(ev.pooled), dtype=float)

    y_tr = np.asarray(tr.labels, dtype=int)
    p_ev = np.argmax(ev.logits, axis=1).astype(int)
    n_classes = d_tr_all.shape[1]

    d_tr = d_tr_all[np.arange(y_tr.shape[0]), y_tr]
    d_ev = d_ev_all[np.arange(p_ev.shape[0]), p_ev]

    global_gate = float(np.quantile(e_tr, 0.70))
    global_high = float(np.quantile(e_tr, 0.98))
    global_span = max(global_high - global_gate, 1e-6)
    global_sorted_d = np.sort(d_tr)

    gates = np.full(n_classes, global_gate, dtype=float)
    spans = np.full(n_classes, global_span, dtype=float)
    sorted_ds = [global_sorted_d] * n_classes

    for c in range(n_classes):
        mask = y_tr == c
        if np.sum(mask) >= 8:
            ec = e_tr[mask]
            dc = d_tr[mask]
            gate = float(np.quantile(ec, 0.70))
            high = float(np.quantile(ec, 0.98))
            gates[c] = gate
            spans[c] = max(high - gate, 1e-6)
            sorted_ds[c] = np.sort(dc)

    out = e_ev.copy()
    for c in range(n_classes):
        mask = p_ev == c
        if not np.any(mask):
            continue

        uncertain = mask & (e_ev > gates[c])
        if not np.any(uncertain):
            continue

        sd = sorted_ds[c]
        rank = np.searchsorted(sd, d_ev[uncertain], side="right") / float(sd.size)
        rank = np.clip(rank, 0.0, 1.0)

        excess = np.maximum(e_ev[uncertain] - gates[c], 0.0)
        energy_tail = 1.0 - np.exp(-excess / spans[c])
        out[uncertain] = gates[c] + spans[c] * (0.90 * rank + 0.10 * energy_tail)

    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
