import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Bottleneck and pooled representations capture near-intent and broad-domain novelty at "
    "different scales. A train-fitted upper-tail excess transform emphasizes unusually weak "
    "support in either view, while inverse-variability weights prevent one density from dominating."
)
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0069", "m_0250"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    gb = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    gp = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_train = tr.feats.shape[0]
    n_eval = ev.feats.shape[0]
    k = tr.logits.shape[1]

    db_all_train = gb.class_mahalanobis_sq(tr.feats)
    dp_all_train = gp.class_mahalanobis_sq(tr.pooled)
    db_all_eval = gb.class_mahalanobis_sq(ev.feats)
    dp_all_eval = gp.class_mahalanobis_sq(ev.pooled)

    db_train = db_all_train[np.arange(n_train), train_pred]
    dp_train = dp_all_train[np.arange(n_train), train_pred]
    db_eval = db_all_eval[np.arange(n_eval), eval_pred]
    dp_eval = dp_all_eval[np.arange(n_eval), eval_pred]

    q75_b_global, q90_b_global = np.quantile(db_train, [0.75, 0.90])
    q75_p_global, q90_p_global = np.quantile(dp_train, [0.75, 0.90])

    tb_eval = np.empty(n_eval, dtype=float)
    tp_eval = np.empty(n_eval, dtype=float)
    tb_train = np.empty(n_train, dtype=float)
    tp_train = np.empty(n_train, dtype=float)

    for c in range(k):
        members = train_pred == c
        if np.sum(members) >= 12:
            q75_b, q90_b = np.quantile(db_train[members], [0.75, 0.90])
            q75_p, q90_p = np.quantile(dp_train[members], [0.75, 0.90])
        else:
            q75_b, q90_b = q75_b_global, q90_b_global
            q75_p, q90_p = q75_p_global, q90_p_global

        scale_b = max(float(q90_b - q75_b), 1e-6)
        scale_p = max(float(q90_p - q75_p), 1e-6)

        train_mask = train_pred == c
        eval_mask = eval_pred == c

        tb_train[train_mask] = np.log1p(
            np.maximum((db_train[train_mask] - q75_b) / scale_b, 0.0)
        )
        tp_train[train_mask] = np.log1p(
            np.maximum((dp_train[train_mask] - q75_p) / scale_p, 0.0)
        )
        tb_eval[eval_mask] = np.log1p(
            np.maximum((db_eval[eval_mask] - q75_b) / scale_b, 0.0)
        )
        tp_eval[eval_mask] = np.log1p(
            np.maximum((dp_eval[eval_mask] - q75_p) / scale_p, 0.0)
        )

    scale_b = max(float(np.mean(tb_train)), 1e-6)
    scale_p = max(float(np.mean(tp_train)), 1e-6)
    weights = np.array([1.0 / scale_b, 1.0 / scale_p], dtype=float)
    weights /= np.sum(weights)

    return weights[0] * tb_eval + weights[1] * tp_eval
