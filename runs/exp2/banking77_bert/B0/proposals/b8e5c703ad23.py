import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often have an unusually small separation between their most likely and runner-up intents; class-conditional train calibration distinguishes genuinely ambiguous OOS predictions from intents that naturally have lower margins."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    z_tr = np.asarray(tr.logits, dtype=float)
    z_ev = np.asarray(ev.logits, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)
    k = z_tr.shape[1]

    def margin(z):
        top2 = np.partition(z, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    m_tr = margin(z_tr)
    m_ev = margin(z_ev)
    global_med = float(np.median(m_tr))
    global_mad = float(np.median(np.abs(m_tr - global_med)))
    global_scale = max(1.4826 * global_mad, 1e-4)

    centers = np.full(k, global_med, dtype=float)
    scales = np.full(k, global_scale, dtype=float)
    for c in range(k):
        vals = m_tr[labels == c]
        if vals.size >= 5:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            weight = vals.size / (vals.size + 20.0)
            centers[c] = weight * med + (1.0 - weight) * global_med
            scales[c] = max(
                weight * 1.4826 * mad + (1.0 - weight) * global_scale,
                1e-4,
            )

    pred = np.argmax(z_ev, axis=1)
    score_out = (centers[pred] - m_ev) / scales[pred]
    return np.nan_to_num(score_out, nan=0.0, posinf=1e6, neginf=-1e6)
