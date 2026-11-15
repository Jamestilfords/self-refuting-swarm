import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and pooled representations capture complementary novelty: the bottleneck is sensitive to deviations near learned intent boundaries, while pooled encoder features capture broader semantic shifts. Predicted-intent-conditioned density surprises are standardized on training data and combined with train-fitted weights based on how well each space distinguishes training prediction errors."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_feat = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_pool = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    tr_pred = np.argmax(tr.logits, axis=1).astype(int)
    ev_pred = np.argmax(ev.logits, axis=1).astype(int)

    tr_feat_all = np.asarray(g_feat.class_mahalanobis_sq(tr.feats), dtype=float)
    tr_pool_all = np.asarray(g_pool.class_mahalanobis_sq(tr.pooled), dtype=float)
    ev_feat_all = np.asarray(g_feat.class_mahalanobis_sq(ev.feats), dtype=float)
    ev_pool_all = np.asarray(g_pool.class_mahalanobis_sq(ev.pooled), dtype=float)

    row_tr = np.arange(tr_pred.shape[0])
    row_ev = np.arange(ev_pred.shape[0])
    tr_feat = tr_feat_all[row_tr, tr_pred]
    tr_pool = tr_pool_all[row_tr, tr_pred]
    ev_feat = ev_feat_all[row_ev, ev_pred]
    ev_pool = ev_pool_all[row_ev, ev_pred]

    k = tr.logits.shape[1]

    def conditional_z(train_values, train_classes, eval_values, eval_classes):
        centers = np.zeros(k, dtype=float)
        scales = np.ones(k, dtype=float)

        global_center = float(np.median(train_values))
        global_scale = float(np.median(np.abs(train_values - global_center)) * 1.4826)
        if not np.isfinite(global_scale) or global_scale < 1e-6:
            global_scale = float(np.std(train_values))
        if not np.isfinite(global_scale) or global_scale < 1e-6:
            global_scale = 1.0

        for c in range(k):
            values = train_values[train_classes == c]
            if values.size >= 5:
                center = float(np.median(values))
                scale = float(np.median(np.abs(values - center)) * 1.4826)
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = float(np.std(values))
                if not np.isfinite(scale) or scale < 1e-6:
                    scale = global_scale
                centers[c] = center
                scales[c] = scale
            else:
                centers[c] = global_center
                scales[c] = global_scale

        train_z = (train_values - centers[train_classes]) / scales[train_classes]
        eval_z = (eval_values - centers[eval_classes]) / scales[eval_classes]
        return train_z, eval_z

    tr_z_feat, ev_z_feat = conditional_z(tr_feat, tr_pred, ev_feat, ev_pred)
    tr_z_pool, ev_z_pool = conditional_z(tr_pool, tr_pred, ev_pool, ev_pred)

    correct = tr_pred == tr.labels

    def proxy_weight(z):
        if np.any(correct) and np.any(~correct):
            gap = float(np.mean(z[~correct]) - np.mean(z[correct]))
            spread = float(np.std(z))
            if not np.isfinite(spread) or spread < 1e-6:
                spread = 1.0
            return max(gap / spread, 0.0)
        return 0.0

    rel_feat = proxy_weight(tr_z_feat)
    rel_pool = proxy_weight(tr_z_pool)

    total = rel_feat + rel_pool
    if total < 1e-8:
        w_feat, w_pool = 0.5, 0.5
    else:
        w_feat = 0.2 + 0.6 * rel_feat / total
        w_pool = 0.2 + 0.6 * rel_pool / total

    values = w_feat * ev_z_feat + w_pool * ev_z_pool
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6)
