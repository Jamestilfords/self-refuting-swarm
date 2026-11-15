import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and pooled representations capture different forms of unsupported input, but only "
    "unusually large class-conditional density deviations should dominate. A train-fitted soft tail "
    "transform emphasizes deviations beyond each intent's ordinary support before additive fusion."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0159"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    labels = np.asarray(tr.labels, dtype=int)
    n_classes = tr.logits.shape[1]

    g_port = ctx.h.gda(tr.feats, labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, labels, cov="tied")

    train_port_all = g_port.class_mahalanobis_sq(tr.feats)
    train_full_all = g_full.class_mahalanobis_sq(tr.pooled)
    eval_port_all = g_port.class_mahalanobis_sq(ev.feats)
    eval_full_all = g_full.class_mahalanobis_sq(ev.pooled)

    row_train = np.arange(labels.size)
    train_port = train_port_all[row_train, labels]
    train_full = train_full_all[row_train, labels]

    q_port = np.zeros(n_classes)
    q_full = np.zeros(n_classes)
    s_port = np.ones(n_classes)
    s_full = np.ones(n_classes)
    reliability_port = np.ones(n_classes)
    reliability_full = np.ones(n_classes)

    global_q_port = np.quantile(train_port, 0.80)
    global_q_full = np.quantile(train_full, 0.80)
    global_s_port = np.quantile(
        np.abs(train_port - np.median(train_port)), 0.75
    ) + 1e-6
    global_s_full = np.quantile(
        np.abs(train_full - np.median(train_full)), 0.75
    ) + 1e-6

    nearest_port = np.argmin(train_port_all, axis=1)
    nearest_full = np.argmin(train_full_all, axis=1)

    for c in range(n_classes):
        mask = labels == c
        dp = train_port[mask]
        df = train_full[mask]

        if dp.size >= 4:
            q_port[c] = np.quantile(dp, 0.80)
            q_full[c] = np.quantile(df, 0.80)

            ep = dp[dp >= q_port[c]] - q_port[c]
            ef = df[df >= q_full[c]] - q_full[c]
            s_port[c] = np.mean(ep) + 1e-6 if ep.size else global_s_port
            s_full[c] = np.mean(ef) + 1e-6 if ef.size else global_s_full
        else:
            q_port[c] = global_q_port
            q_full[c] = global_q_full
            s_port[c] = global_s_port
            s_full[c] = global_s_full

        count = max(int(mask.sum()), 1)
        agree_port = np.mean(nearest_port[mask] == c) if mask.any() else 0.0
        agree_full = np.mean(nearest_full[mask] == c) if mask.any() else 0.0
        shrink = count / (count + 20.0)
        reliability_port[c] = shrink * agree_port + (1.0 - shrink) * 0.5
        reliability_full[c] = shrink * agree_full + (1.0 - shrink) * 0.5

    weights_sum = reliability_port + reliability_full + 1e-8
    w_port = reliability_port / weights_sum
    w_full = reliability_full / weights_sum

    pred = np.argmax(ev.logits, axis=1).astype(int)
    rows = np.arange(pred.size)
    d_port = eval_port_all[rows, pred]
    d_full = eval_full_all[rows, pred]

    z_port = (d_port - q_port[pred]) / s_port[pred]
    z_full = (d_full - q_full[pred]) / s_full[pred]

    tail_port = np.log1p(np.maximum(z_port, 0.0))
    tail_full = np.log1p(np.maximum(z_full, 0.0))

    return (
        w_port[pred] * (tail_port + 0.02 * z_port)
        + w_full[pred] * (tail_full + 0.02 * z_full)
    )
