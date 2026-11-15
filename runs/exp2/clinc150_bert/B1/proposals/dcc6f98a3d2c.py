import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Predicted-class Mahalanobis deviations in bottleneck space capture near-OOS intent mismatches, while encoder-space "
    "deviations capture broader semantic shifts. Class-conditional empirical tail calibration and covariance-derived "
    "train-only weights make the two density signals complementary rather than a near-duplicate linear distance sum."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_port_all = g_port.class_mahalanobis_sq(tr.feats)
    train_full_all = g_full.class_mahalanobis_sq(tr.pooled)
    eval_port_all = g_port.class_mahalanobis_sq(ev.feats)
    eval_full_all = g_full.class_mahalanobis_sq(ev.pooled)

    k_port = train_port_all.shape[1]
    k_full = train_full_all.shape[1]
    train_pred_port = np.clip(train_pred, 0, k_port - 1)
    train_pred_full = np.clip(train_pred, 0, k_full - 1)
    eval_pred_port = np.clip(eval_pred, 0, k_port - 1)
    eval_pred_full = np.clip(eval_pred, 0, k_full - 1)

    train_port = train_port_all[np.arange(len(tr.feats)), train_pred_port]
    train_full = train_full_all[np.arange(len(tr.pooled)), train_pred_full]
    eval_port = eval_port_all[np.arange(len(ev.feats)), eval_pred_port]
    eval_full = eval_full_all[np.arange(len(ev.pooled)), eval_pred_full]

    def calibrated_tail(train_values, train_classes, eval_values, eval_classes):
        global_sorted = np.sort(train_values)
        train_out = np.empty_like(train_values, dtype=float)
        eval_out = np.empty_like(eval_values, dtype=float)

        for c in np.unique(np.concatenate((train_classes, eval_classes))):
            train_mask = train_classes == c
            eval_mask = eval_classes == c
            reference = train_values[train_mask]
            if reference.size < 8:
                reference = global_sorted
            else:
                reference = np.sort(reference)

            n = float(reference.size)
            if np.any(train_mask):
                ranks = np.searchsorted(reference, train_values[train_mask], side="right")
                tail = (n - ranks + 0.5) / (n + 1.0)
                train_out[train_mask] = -np.log(np.maximum(tail, 1e-12))
            if np.any(eval_mask):
                ranks = np.searchsorted(reference, eval_values[eval_mask], side="right")
                tail = (n - ranks + 0.5) / (n + 1.0)
                eval_out[eval_mask] = -np.log(np.maximum(tail, 1e-12))

        return train_out, eval_out

    train_port_tail, eval_port_tail = calibrated_tail(
        train_port, train_pred_port, eval_port, eval_pred_port
    )
    train_full_tail, eval_full_tail = calibrated_tail(
        train_full, train_pred_full, eval_full, eval_pred_full
    )

    train_signals = np.column_stack((train_port_tail, train_full_tail))
    cov = np.cov(train_signals, rowvar=False)
    cov = np.atleast_2d(cov) + 1e-6 * np.eye(2)
    precision = np.linalg.pinv(cov)
    weights = precision @ np.ones(2)
    weights = np.maximum(weights, 0.05)
    weights /= weights.sum()

    scores = weights[0] * eval_port_tail + weights[1] * eval_full_tail
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
