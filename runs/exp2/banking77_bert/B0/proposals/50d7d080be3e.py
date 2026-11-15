import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Predicted-intent logit geometry and predicted-intent prototype alignment capture complementary "
    "forms of support. Train-fitted empirical percentile calibration puts their class-conditional "
    "atypicality scores on a common scale before combining them."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0143", "m_0167", "m_0169"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12
    k = tr.logits.shape[1]

    def logit_features(logits):
        z = logits - np.max(logits, axis=1, keepdims=True)
        p = np.exp(z)
        p /= np.sum(p, axis=1, keepdims=True)
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        entropy = -np.sum(p * np.log(np.maximum(p, eps)), axis=1)
        centered_norm = np.sqrt(np.mean((logits - logits.mean(axis=1, keepdims=True)) ** 2, axis=1))
        return np.column_stack((margin, entropy, centered_norm))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    lf_train = logit_features(tr.logits)
    lf_eval = logit_features(ev.logits)

    global_med = np.median(lf_train, axis=0)
    global_scale = np.median(np.abs(lf_train - global_med), axis=0) * 1.4826 + eps

    logit_med = np.tile(global_med, (k, 1))
    logit_scale = np.tile(global_scale, (k, 1))
    for c in range(k):
        mask = train_pred == c
        if np.sum(mask) >= 5:
            values = lf_train[mask]
            med = np.median(values, axis=0)
            scale = np.median(np.abs(values - med), axis=0) * 1.4826 + eps
            logit_med[c] = med
            logit_scale[c] = scale

    train_logit_raw = np.sum(
        ((lf_train - logit_med[train_pred]) / logit_scale[train_pred]) ** 2, axis=1
    )
    eval_logit_raw = np.sum(
        ((lf_eval - logit_med[eval_pred]) / logit_scale[eval_pred]) ** 2, axis=1
    )

    x_train = tr.pooled / np.maximum(np.linalg.norm(tr.pooled, axis=1, keepdims=True), eps)
    x_eval = ev.pooled / np.maximum(np.linalg.norm(ev.pooled, axis=1, keepdims=True), eps)
    global_proto = np.mean(x_train, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), eps)

    prototypes = np.tile(global_proto, (k, 1))
    for c in range(k):
        mask = train_pred == c
        if np.sum(mask) >= 2:
            proto = np.mean(x_train[mask], axis=0)
            norm = np.linalg.norm(proto)
            if norm > eps:
                prototypes[c] = proto / norm

    train_feature_raw = 1.0 - np.sum(x_train * prototypes[train_pred], axis=1)
    eval_feature_raw = 1.0 - np.sum(x_eval * prototypes[eval_pred], axis=1)

    logit_ref = np.sort(train_logit_raw)
    feature_ref = np.sort(train_feature_raw)
    logit_pct = np.searchsorted(logit_ref, eval_logit_raw, side="right") / (len(logit_ref) + 1.0)
    feature_pct = np.searchsorted(feature_ref, eval_feature_raw, side="right") / (len(feature_ref) + 1.0)

    return 0.5 * logit_pct + 0.5 * feature_pct
