import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier-selected class density is calibrated separately for bottleneck and encoder views, "
    "so an input is anomalous when its predicted intent lacks support in either near-intent or broad-domain geometry."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0118", "m_0120"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    g_feat = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_pool = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    d_feat_tr = g_feat.class_mahalanobis_sq(tr.feats)
    d_pool_tr = g_pool.class_mahalanobis_sq(tr.pooled)
    d_feat_ev = g_feat.class_mahalanobis_sq(ev.feats)
    d_pool_ev = g_pool.class_mahalanobis_sq(ev.pooled)

    pred_tr = np.clip(pred_tr, 0, d_feat_tr.shape[1] - 1)
    pred_ev = np.clip(pred_ev, 0, d_feat_ev.shape[1] - 1)
    rows_tr = np.arange(len(pred_tr))
    rows_ev = np.arange(len(pred_ev))

    x_feat_tr = d_feat_tr[rows_tr, pred_tr]
    x_pool_tr = d_pool_tr[rows_tr, pred_tr]
    x_feat_ev = d_feat_ev[rows_ev, pred_ev]
    x_pool_ev = d_pool_ev[rows_ev, pred_ev]

    n_classes = d_feat_tr.shape[1]
    med_feat = np.zeros(n_classes)
    med_pool = np.zeros(n_classes)
    scale_feat = np.ones(n_classes)
    scale_pool = np.ones(n_classes)
    w_feat = np.full(n_classes, 0.5)
    w_pool = np.full(n_classes, 0.5)

    global_med_feat = float(np.median(x_feat_tr))
    global_med_pool = float(np.median(x_pool_tr))
    global_scale_feat = float(np.quantile(x_feat_tr, 0.75) - np.quantile(x_feat_tr, 0.25))
    global_scale_pool = float(np.quantile(x_pool_tr, 0.75) - np.quantile(x_pool_tr, 0.25))
    global_scale_feat = max(global_scale_feat, eps)
    global_scale_pool = max(global_scale_pool, eps)

    for c in range(n_classes):
        mask = pred_tr == c
        if np.sum(mask) >= 8:
            a = x_feat_tr[mask]
            b = x_pool_tr[mask]
            med_feat[c] = np.median(a)
            med_pool[c] = np.median(b)
            scale_feat[c] = max(np.quantile(a, 0.75) - np.quantile(a, 0.25), eps)
            scale_pool[c] = max(np.quantile(b, 0.75) - np.quantile(b, 0.25), eps)
        else:
            med_feat[c] = global_med_feat
            med_pool[c] = global_med_pool
            scale_feat[c] = global_scale_feat
            scale_pool[c] = global_scale_pool

        correct = mask & (tr.labels == c)
        wrong = mask & (tr.labels != c)
        if np.sum(correct) >= 4 and np.sum(wrong) >= 4:
            feat_gap = (
                np.mean(x_feat_tr[wrong]) - np.mean(x_feat_tr[correct])
            ) / scale_feat[c]
            pool_gap = (
                np.mean(x_pool_tr[wrong]) - np.mean(x_pool_tr[correct])
            ) / scale_pool[c]
            logits = np.clip(np.array([feat_gap, pool_gap]), -20.0, 20.0)
            logits -= np.max(logits)
            weights = np.exp(logits)
            weights /= np.sum(weights)
            w_feat[c], w_pool[c] = weights

    z_feat = (x_feat_ev - med_feat[pred_ev]) / scale_feat[pred_ev]
    z_pool = (x_pool_ev - med_pool[pred_ev]) / scale_pool[pred_ev]

    tail_feat = 1.0 / (1.0 + np.exp(-np.clip(z_feat, -40.0, 40.0)))
    tail_pool = 1.0 / (1.0 + np.exp(-np.clip(z_pool, -40.0, 40.0)))

    scores = w_feat[pred_ev] * tail_feat + w_pool[pred_ev] * tail_pool
    return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
