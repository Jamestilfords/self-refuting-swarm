import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and encoder densities are calibrated separately within the classifier-predicted intent, so near-OOS and domain-shift deviations are comparable without relying on a global density scale. "
    "Train-fitted reliability weights are constrained to retain both views, preventing either representation space from collapsing the fusion into a duplicate single-density ranking."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0065", "m_0084"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train = -np.asarray(g_port.log_density(tr.feats), dtype=float)
    full_train = -np.asarray(g_full.log_density(tr.pooled), dtype=float)
    port_eval = -np.asarray(g_port.log_density(ev.feats), dtype=float)
    full_eval = -np.asarray(g_full.log_density(ev.pooled), dtype=float)

    pred_train = np.argmax(tr.logits, axis=1)
    pred_eval = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    port_score = np.empty_like(port_eval)
    full_score = np.empty_like(full_eval)
    weights = np.empty((n_classes, 2), dtype=float)

    global_port_med = np.median(port_train)
    global_full_med = np.median(full_train)
    global_port_scale = np.median(np.abs(port_train - global_port_med)) * 1.4826 + 1e-6
    global_full_scale = np.median(np.abs(full_train - global_full_med)) * 1.4826 + 1e-6

    for c in range(n_classes):
        mask = pred_train == c
        if np.sum(mask) < 8:
            p_ref, p_scale = global_port_med, global_port_scale
            f_ref, f_scale = global_full_med, global_full_scale
        else:
            p_values = port_train[mask]
            f_values = full_train[mask]
            p_ref = np.median(p_values)
            f_ref = np.median(f_values)
            p_scale = np.median(np.abs(p_values - p_ref)) * 1.4826 + 1e-6
            f_scale = np.median(np.abs(f_values - f_ref)) * 1.4826 + 1e-6

        p_rel = 1.0 / p_scale
        f_rel = 1.0 / f_scale
        raw = np.array([p_rel, f_rel], dtype=float)
        raw /= np.sum(raw)
        weights[c] = 0.2 + 0.6 * raw

        use = pred_eval == c
        port_score[use] = (port_eval[use] - p_ref) / p_scale
        full_score[use] = (full_eval[use] - f_ref) / f_scale

    scores = (
        weights[pred_eval, 0] * port_score
        + weights[pred_eval, 1] * full_score
    )
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
