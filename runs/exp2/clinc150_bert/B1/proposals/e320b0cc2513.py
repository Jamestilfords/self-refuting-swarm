import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Class-conditional bottleneck and pooled-space density deviations target complementary near-intent and broad semantic shifts. Training-only per-intent robust calibration and a nonlinear tail transform prevent either representation's raw Mahalanobis scale from dominating the additive fusion."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0080"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-6

    g_feat = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_pool = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_feat_all = g_feat.class_mahalanobis_sq(tr.feats)
    train_pool_all = g_pool.class_mahalanobis_sq(tr.pooled)
    row = np.arange(tr.labels.shape[0])

    train_feat = train_feat_all[row, tr.labels]
    train_pool = train_pool_all[row, tr.labels]

    n_classes = train_feat_all.shape[1]
    med_feat = np.zeros(n_classes, dtype=float)
    med_pool = np.zeros(n_classes, dtype=float)
    scale_feat = np.zeros(n_classes, dtype=float)
    scale_pool = np.zeros(n_classes, dtype=float)

    global_feat_scale = max(
        float(np.quantile(train_feat, 0.75) - np.quantile(train_feat, 0.25)),
        eps,
    )
    global_pool_scale = max(
        float(np.quantile(train_pool, 0.75) - np.quantile(train_pool, 0.25)),
        eps,
    )

    for c in range(n_classes):
        mask = tr.labels == c
        x_feat = train_feat[mask]
        x_pool = train_pool[mask]

        if x_feat.size >= 4:
            med_feat[c] = np.median(x_feat)
            scale_feat[c] = np.quantile(x_feat, 0.75) - np.quantile(x_feat, 0.25)
        else:
            med_feat[c] = np.median(train_feat)
            scale_feat[c] = global_feat_scale

        if x_pool.size >= 4:
            med_pool[c] = np.median(x_pool)
            scale_pool[c] = np.quantile(x_pool, 0.75) - np.quantile(x_pool, 0.25)
        else:
            med_pool[c] = np.median(train_pool)
            scale_pool[c] = global_pool_scale

    scale_feat = np.maximum(scale_feat, eps)
    scale_pool = np.maximum(scale_pool, eps)

    train_z_feat = (train_feat - med_feat[tr.labels]) / scale_feat[tr.labels]
    train_z_pool = (train_pool - med_pool[tr.labels]) / scale_pool[tr.labels]

    spread_feat = max(float(np.std(np.arcsinh(train_z_feat))), eps)
    spread_pool = max(float(np.std(np.arcsinh(train_z_pool))), eps)
    w_feat = 1.0 / spread_feat
    w_pool = 1.0 / spread_pool
    weight_sum = w_feat + w_pool
    w_feat /= weight_sum
    w_pool /= weight_sum

    pred = np.argmax(ev.logits, axis=1)
    eval_feat_all = g_feat.class_mahalanobis_sq(ev.feats)
    eval_pool_all = g_pool.class_mahalanobis_sq(ev.pooled)
    erow = np.arange(pred.shape[0])

    eval_feat = eval_feat_all[erow, pred]
    eval_pool = eval_pool_all[erow, pred]

    z_feat = (eval_feat - med_feat[pred]) / scale_feat[pred]
    z_pool = (eval_pool - med_pool[pred]) / scale_pool[pred]

    out = w_feat * np.arcsinh(z_feat) + w_pool * np.arcsinh(z_pool)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
