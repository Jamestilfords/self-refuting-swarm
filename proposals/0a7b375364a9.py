import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Classifier-selected intents should receive compatible diagonal-Gaussian support in both "
    "the pooled encoder representation and the bottleneck representation. A class-calibrated "
    "joint anomaly of these two independently variance-normalized distances catches OOS inputs "
    "that are unsupported in either view or exhibit unusual cross-view disagreement."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8
    k = tr.logits.shape[1]

    def class_diagonal_stats(x):
        global_mean = np.mean(x, axis=0)
        global_var = np.var(x, axis=0) + eps
        means = np.tile(global_mean, (k, 1))
        variances = np.tile(global_var, (k, 1))

        for c in range(k):
            xc = x[tr.labels == c]
            n = xc.shape[0]
            if n:
                mu = np.mean(xc, axis=0)
                var = np.mean((xc - mu) ** 2, axis=0)
                shrink = n / (n + 20.0)
                means[c] = mu
                variances[c] = shrink * var + (1.0 - shrink) * global_var + eps
        return means, variances

    pool_mu, pool_var = class_diagonal_stats(tr.pooled)
    feat_mu, feat_var = class_diagonal_stats(tr.feats)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def selected_distance(x, pred, means, variances):
        residual = x - means[pred]
        return np.sum((residual * residual) / variances[pred], axis=1)

    train_pool_d = selected_distance(tr.pooled, train_pred, pool_mu, pool_var)
    train_feat_d = selected_distance(tr.feats, train_pred, feat_mu, feat_var)
    eval_pool_d = selected_distance(ev.pooled, eval_pred, pool_mu, pool_var)
    eval_feat_d = selected_distance(ev.feats, eval_pred, feat_mu, feat_var)

    train_joint = np.column_stack((np.log1p(train_pool_d), np.log1p(train_feat_d)))
    eval_joint = np.column_stack((np.log1p(eval_pool_d), np.log1p(eval_feat_d)))

    global_center = np.mean(train_joint, axis=0)
    centered = train_joint - global_center
    global_cov = centered.T @ centered / max(train_joint.shape[0], 1)
    ridge = max(float(np.trace(global_cov)) * 1e-5, 1e-6)
    global_cov = global_cov + ridge * np.eye(2)

    centers = np.tile(global_center, (k, 1))
    precisions = np.empty((k, 2, 2))
    for c in range(k):
        zc = train_joint[train_pred == c]
        n = zc.shape[0]
        if n:
            center = np.mean(zc, axis=0)
            dz = zc - center
            cov_c = dz.T @ dz / n
            shrink = n / (n + 25.0)
            cov = shrink * cov_c + (1.0 - shrink) * global_cov
            centers[c] = center
        else:
            cov = global_cov
        cov = cov + ridge * np.eye(2)
        precisions[c] = np.linalg.pinv(cov)

    delta = eval_joint - centers[eval_pred]
    score_values = np.einsum("ni,nij,nj->n", delta, precisions[eval_pred], delta)
    return np.maximum(score_values, 0.0)
