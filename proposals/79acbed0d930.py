import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Near-OOS inputs can depart from class structure in the bottleneck while far-domain inputs "
    "depart in pooled encoder space. Predicted-intent-specific train-tail calibration and "
    "train-fitted reliability weights combine these complementary density deviations."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0003", "m_0084", "m_0118"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=np.int64)
    n_classes = tr.logits.shape[1]

    g_port = ctx.h.gda(tr.feats, labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, labels, cov="tied")

    port_train_all = g_port.class_mahalanobis_sq(tr.feats)
    full_train_all = g_full.class_mahalanobis_sq(tr.pooled)
    port_eval_all = g_port.class_mahalanobis_sq(ev.feats)
    full_eval_all = g_full.class_mahalanobis_sq(ev.pooled)

    train_idx = np.arange(labels.shape[0])
    pred_eval = np.argmax(ev.logits, axis=1).astype(np.int64)
    eval_idx = np.arange(pred_eval.shape[0])

    port_train = port_train_all[train_idx, labels]
    full_train = full_train_all[train_idx, labels]
    port_eval = port_eval_all[eval_idx, pred_eval]
    full_eval = full_eval_all[eval_idx, pred_eval]

    tail_port = np.empty_like(port_eval, dtype=float)
    tail_full = np.empty_like(full_eval, dtype=float)
    w_port = np.empty(n_classes, dtype=float)
    w_full = np.empty(n_classes, dtype=float)

    global_port = np.sort(port_train)
    global_full = np.sort(full_train)
    global_v_port = np.var(np.log1p(np.maximum(port_train, 0.0))) + 1e-6
    global_v_full = np.var(np.log1p(np.maximum(full_train, 0.0))) + 1e-6

    for c in range(n_classes):
        mask_train = labels == c
        mask_eval = pred_eval == c

        if np.sum(mask_train) >= 3:
            port_ref = np.sort(port_train[mask_train])
            full_ref = np.sort(full_train[mask_train])
            var_port = np.var(np.log1p(np.maximum(port_train[mask_train], 0.0))) + 1e-6
            var_full = np.var(np.log1p(np.maximum(full_train[mask_train], 0.0))) + 1e-6
        else:
            port_ref = global_port
            full_ref = global_full
            var_port = global_v_port
            var_full = global_v_full

        w_port[c] = 1.0 / var_port
        w_full[c] = 1.0 / var_full

        if np.any(mask_eval):
            tail_port[mask_eval] = (
                np.searchsorted(port_ref, port_eval[mask_eval], side="right")
                / float(port_ref.shape[0])
            )
            tail_full[mask_eval] = (
                np.searchsorted(full_ref, full_eval[mask_eval], side="right")
                / float(full_ref.shape[0])
            )

    wp = w_port[pred_eval]
    wf = w_full[pred_eval]
    scores = (wp * tail_port + wf * tail_full) / (wp + wf)
    scores += 1e-8 * (port_eval / (1.0 + np.abs(port_eval)))

    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
