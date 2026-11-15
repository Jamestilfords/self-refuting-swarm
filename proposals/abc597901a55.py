import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet allocation has a characteristic winner share and rival-concentration pattern for each predicted intent. "
    "OOS queries can induce unusually diffuse or unusually concentrated evidential competition even when total evidence, energy, and vacuity appear typical."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0204"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    def allocation_features(alpha, logits):
        pred = np.argmax(logits, axis=1)
        total = np.sum(alpha, axis=1)
        winner = alpha[np.arange(alpha.shape[0]), pred] / total
        tail_sum = np.maximum(total - alpha[np.arange(alpha.shape[0]), pred], 1e-12)
        tail_sq = np.sum(alpha * alpha, axis=1) - alpha[np.arange(alpha.shape[0]), pred] ** 2
        tail_hhi = tail_sq / (tail_sum * tail_sum)
        return pred, np.column_stack((winner, tail_hhi))

    p_tr, f_tr = allocation_features(a_tr, tr.logits)
    p_ev, f_ev = allocation_features(a_ev, ev.logits)
    k = tr.logits.shape[1]

    global_med = np.median(f_tr, axis=0)
    global_mad = np.median(np.abs(f_tr - global_med), axis=0)
    global_std = np.std(f_tr, axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 0.05 * global_std)
    global_scale = np.maximum(global_scale, 1e-6)

    med = np.tile(global_med, (k, 1))
    scale = np.tile(global_scale, (k, 1))
    for c in range(k):
        rows = f_tr[p_tr == c]
        if rows.shape[0] >= 5:
            center = np.median(rows, axis=0)
            mad = np.median(np.abs(rows - center), axis=0)
            spread = np.std(rows, axis=0)
            med[c] = center
            scale[c] = np.maximum(np.maximum(1.4826 * mad, 0.05 * spread), 1e-6)

    z = np.abs((f_ev - med[p_ev]) / scale[p_ev])
    out = z[:, 0] + z[:, 1]
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=0.0)
