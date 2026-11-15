import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Classifier-conditioned Mahalanobis distances make each representation measure deviation from the intent the model assigns, rather than allowing an unrelated class to explain the sample. Robust per-predicted-intent calibration and train-fitted inverse-variance weights combine bottleneck sensitivity to nearby held-out intents with encoder sensitivity to far-domain inputs."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    port_train_all = np.asarray(g_port.class_mahalanobis_sq(tr.feats), dtype=float)
    full_train_all = np.asarray(g_full.class_mahalanobis_sq(tr.pooled), dtype=float)
    port_eval_all = np.asarray(g_port.class_mahalanobis_sq(ev.feats), dtype=float)
    full_eval_all = np.asarray(g_full.class_mahalanobis_sq(ev.pooled), dtype=float)

    n_classes = min(
        port_train_all.shape[1],
        full_train_all.shape[1],
        tr.logits.shape[1],
        ev.logits.shape[1],
    )

    train_pred = np.argmax(tr.logits[:, :n_classes], axis=1)
    eval_pred = np.argmax(ev.logits[:, :n_classes], axis=1)

    row_train = np.arange(tr.feats.shape[0])
    row_eval = np.arange(ev.feats.shape[0])

    port_train = port_train_all[row_train, train_pred]
    full_train = full_train_all[row_train, train_pred]
    port_eval = port_eval_all[row_eval, eval_pred]
    full_eval = full_eval_all[row_eval, eval_pred]

    def class_calibrate(train_values, eval_values):
        global_med = np.median(train_values)
        global_scale = np.median(np.abs(train_values - global_med)) * 1.4826
        if not np.isfinite(global_scale) or global_scale < 1e-8:
            global_scale = np.std(train_values) + 1e-8

        medians = np.full(n_classes, global_med, dtype=float)
        scales = np.full(n_classes, global_scale, dtype=float)

        for cls in range(n_classes):
            values = train_values[train_pred == cls]
            if values.size >= 8:
                med = np.median(values)
                scale = np.median(np.abs(values - med)) * 1.4826
                if np.isfinite(scale) and scale >= 1e-8:
                    medians[cls] = med
                    scales[cls] = scale

        train_z = (train_values - medians[train_pred]) / scales[train_pred]
        eval_z = (eval_values - medians[eval_pred]) / scales[eval_pred]

        return np.arcsinh(train_z), np.arcsinh(eval_z)

    port_train_z, port_eval_z = class_calibrate(port_train, port_eval)
    full_train_z, full_eval_z = class_calibrate(full_train, full_eval)

    port_var = np.var(port_train_z)
    full_var = np.var(full_train_z)
    w_port = 1.0 / max(port_var, 1e-8)
    w_full = 1.0 / max(full_var, 1e-8)
    weight_sum = w_port + w_full

    scores = (w_port * port_eval_z + w_full * full_eval_z) / weight_sum
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
