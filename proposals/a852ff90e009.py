import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "OOS inputs can have both an atypical ranked logit competition pattern and weak "
    "support from the feature prototype of the classifier-selected intent. "
    "Class-conditional train-tail maps place these complementary signals on a common scale."
)
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = ["m_0088", "m_0099", "m_0100"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    eps = 1e-8

    def spectrum(z):
        centered = z - z.mean(axis=1, keepdims=True)
        centered = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), eps)
        ordered = np.sort(centered, axis=1)[:, ::-1]
        return ordered[:, :1] - ordered[:, 1:]

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    tr_spec = spectrum(tr.logits)
    ev_spec = spectrum(ev.logits)

    global_mean = tr_spec.mean(axis=0)
    global_var = tr_spec.var(axis=0) + 1e-4
    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))

    for c in range(k):
        idx = tr_pred == c
        if idx.sum() >= 4:
            means[c] = tr_spec[idx].mean(axis=0)
            variances[c] = tr_spec[idx].var(axis=0) + 1e-4

    tr_logit_shape = np.sum(
        (tr_spec - means[tr_pred]) ** 2 / variances[tr_pred], axis=1
    )
    ev_logit_shape = np.sum(
        (ev_spec - means[ev_pred]) ** 2 / variances[ev_pred], axis=1
    )

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    tr_all_dist = g.class_mahalanobis_sq(tr.pooled)
    ev_all_dist = g.class_mahalanobis_sq(ev.pooled)
    tr_feature = tr_all_dist[np.arange(len(tr_pred)), tr_pred]
    ev_feature = ev_all_dist[np.arange(len(ev_pred)), ev_pred]

    tr_msp = ctx.h.msp(tr.logits)
    ev_msp = ctx.h.msp(ev.logits)

    def upper_tail(train_values, values):
        ordered = np.sort(np.asarray(train_values, dtype=float))
        return np.searchsorted(ordered, values, side="right") / float(len(ordered) + 1)

    shape_tail = np.empty(len(ev_pred), dtype=float)
    feature_tail = np.empty(len(ev_pred), dtype=float)

    for c in range(k):
        train_idx = tr_pred == c
        eval_idx = ev_pred == c
        if not np.any(eval_idx):
            continue
        if train_idx.sum() >= 8:
            shape_tail[eval_idx] = upper_tail(
                tr_logit_shape[train_idx], ev_logit_shape[eval_idx]
            )
            feature_tail[eval_idx] = upper_tail(
                tr_feature[train_idx], ev_feature[eval_idx]
            )
        else:
            shape_tail[eval_idx] = upper_tail(tr_logit_shape, ev_logit_shape[eval_idx])
            feature_tail[eval_idx] = upper_tail(tr_feature, ev_feature[eval_idx])

    msp_tail = upper_tail(tr_msp, ev_msp)
    logit_tail = 0.5 * shape_tail + 0.5 * msp_tail

    return np.sqrt(np.maximum(logit_tail, eps) * np.maximum(feature_tail, eps))
