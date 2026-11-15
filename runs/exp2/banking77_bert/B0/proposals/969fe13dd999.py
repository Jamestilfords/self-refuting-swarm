import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Near-OOS inputs can be unsupported by the predicted intent in bottleneck space, while far-OOS domain shifts can be unsupported in encoder space. Class-conditional train-calibrated density tails make the two distances comparable before their train-fitted additive fusion."
SOURCE = "https://arxiv.org/abs/2106.09022"
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0115", "m_0130"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_classes = tr.logits.shape[1]

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    port_train_all = np.asarray(g_port.class_mahalanobis_sq(tr.feats), dtype=float)
    full_train_all = np.asarray(g_full.class_mahalanobis_sq(tr.pooled), dtype=float)
    port_eval_all = np.asarray(g_port.class_mahalanobis_sq(ev.feats), dtype=float)
    full_eval_all = np.asarray(g_full.class_mahalanobis_sq(ev.pooled), dtype=float)

    train_idx = np.arange(tr.logits.shape[0])
    eval_idx = np.arange(ev.logits.shape[0])
    port_train = port_train_all[train_idx, train_pred]
    full_train = full_train_all[train_idx, train_pred]
    port_eval = port_eval_all[eval_idx, eval_pred]
    full_eval = full_eval_all[eval_idx, eval_pred]

    def conditional_tail(train_values, train_classes, values, value_classes):
        result = np.empty(values.shape[0], dtype=float)
        global_sorted = np.sort(train_values)
        for c in range(n_classes):
            mask_train = train_classes == c
            mask_eval = value_classes == c
            if not np.any(mask_eval):
                continue
            reference = train_values[mask_train]
            if reference.size < 8:
                reference = global_sorted
            else:
                reference = np.sort(reference)
            ranks = np.searchsorted(reference, values[mask_eval], side="right")
            result[mask_eval] = ranks / (reference.size + 1.0)
        return result

    port_tail_train = conditional_tail(port_train, train_pred, port_train, train_pred)
    full_tail_train = conditional_tail(full_train, train_pred, full_train, train_pred)
    port_tail_eval = conditional_tail(port_train, train_pred, port_eval, eval_pred)
    full_tail_eval = conditional_tail(full_train, train_pred, full_eval, eval_pred)

    train_components = np.column_stack([port_tail_train, full_tail_train])
    variances = np.var(train_components, axis=0)
    weights = 1.0 / np.maximum(variances, 1e-8)
    weights = weights / np.sum(weights)

    scores = weights[0] * port_tail_eval + weights[1] * full_tail_eval
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
