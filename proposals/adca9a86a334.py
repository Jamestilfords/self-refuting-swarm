import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier-selected class-conditional distances measure whether the representation supports the intent the model predicts, "
    "while separate bottleneck and encoder views capture near-intent and far-domain deviations. Train-fitted robust scaling and "
    "saturating additive fusion prevent either representation scale from dominating."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0146"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    port_tr_all = np.nan_to_num(
        g_port.class_mahalanobis_sq(tr.feats),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    full_tr_all = np.nan_to_num(
        g_full.class_mahalanobis_sq(tr.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    port_ev_all = np.nan_to_num(
        g_port.class_mahalanobis_sq(ev.feats),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    full_ev_all = np.nan_to_num(
        g_full.class_mahalanobis_sq(ev.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )

    max_class = min(
        port_tr_all.shape[1],
        full_tr_all.shape[1],
        port_ev_all.shape[1],
        full_ev_all.shape[1],
    ) - 1
    pred_tr = np.clip(pred_tr, 0, max_class)
    pred_ev = np.clip(pred_ev, 0, max_class)

    rows_tr = np.arange(tr.logits.shape[0])
    rows_ev = np.arange(ev.logits.shape[0])
    port_tr = port_tr_all[rows_tr, pred_tr]
    full_tr = full_tr_all[rows_tr, pred_tr]
    port_ev = port_ev_all[rows_ev, pred_ev]
    full_ev = full_ev_all[rows_ev, pred_ev]

    def class_robust_scale(train_values, eval_values):
        global_med = np.median(train_values)
        global_mad = np.median(np.abs(train_values - global_med))
        global_scale = max(1.4826 * global_mad, 1e-6)

        med = np.full(max_class + 1, global_med, dtype=float)
        scale = np.full(max_class + 1, global_scale, dtype=float)

        for c in range(max_class + 1):
            values = train_values[pred_tr == c]
            if values.size >= 8:
                center = np.median(values)
                mad = np.median(np.abs(values - center))
                med[c] = center
                scale[c] = max(1.4826 * mad, 0.15 * global_scale, 1e-6)

        train_z = (train_values - med[pred_tr]) / scale[pred_tr]
        eval_z = (eval_values - med[pred_ev]) / scale[pred_ev]
        return train_z, eval_z

    port_tr_z, port_ev_z = class_robust_scale(port_tr, port_ev)
    full_tr_z, full_ev_z = class_robust_scale(full_tr, full_ev)

    def upper_excess(z):
        return np.log1p(np.exp(np.clip(z, -30.0, 30.0)))

    port_train_signal = upper_excess(port_tr_z)
    full_train_signal = upper_excess(full_tr_z)
    port_eval_signal = upper_excess(port_ev_z)
    full_eval_signal = upper_excess(full_ev_z)

    port_var = max(np.var(port_train_signal), 1e-6)
    full_var = max(np.var(full_train_signal), 1e-6)
    weights = np.array([1.0 / port_var, 1.0 / full_var], dtype=float)
    weights /= weights.sum()

    scores = weights[0] * port_eval_signal + weights[1] * full_eval_signal
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
