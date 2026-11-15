import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Energy and pooled-space Mahalanobis distance provide complementary uncertainty signals. "
    "Predicted-intent-conditioned robust scaling learned only from training data makes their magnitudes comparable while preserving intent-specific in-scope variability."
)
SOURCE = ""
PARENTS = ["energy", "d_full"]
MEMORY_USED = ["m_0093"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_energy = np.asarray(ctx.h.energy(tr.logits), dtype=float)
    ev_energy = np.asarray(ctx.h.energy(ev.logits), dtype=float)
    tr_dist = np.asarray(g.min_mahalanobis_sq(tr.pooled), dtype=float)
    ev_dist = np.asarray(g.min_mahalanobis_sq(ev.pooled), dtype=float)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_e_med = np.median(tr_energy)
    global_d_med = np.median(tr_dist)
    global_e_scale = np.median(np.abs(tr_energy - global_e_med)) * 1.4826
    global_d_scale = np.median(np.abs(tr_dist - global_d_med)) * 1.4826
    global_e_scale = max(float(global_e_scale), 1e-6)
    global_d_scale = max(float(global_d_scale), 1e-6)

    e_med = np.full(k, global_e_med, dtype=float)
    d_med = np.full(k, global_d_med, dtype=float)
    e_scale = np.full(k, global_e_scale, dtype=float)
    d_scale = np.full(k, global_d_scale, dtype=float)

    for c in range(k):
        mask = tr_pred == c
        if np.sum(mask) >= 8:
            ce = tr_energy[mask]
            cd = tr_dist[mask]
            em = np.median(ce)
            dm = np.median(cd)
            es = np.median(np.abs(ce - em)) * 1.4826
            ds = np.median(np.abs(cd - dm)) * 1.4826
            e_med[c] = em
            d_med[c] = dm
            e_scale[c] = max(float(es), 1e-6)
            d_scale[c] = max(float(ds), 1e-6)

    energy_z = (ev_energy - e_med[ev_pred]) / e_scale[ev_pred]
    distance_z = (ev_dist - d_med[ev_pred]) / d_scale[ev_pred]
    scores = 0.5 * energy_z + 0.5 * distance_z
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
