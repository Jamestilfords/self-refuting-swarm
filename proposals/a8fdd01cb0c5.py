import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "When the classifier's predicted intent and its strongest logit rival are compared geometrically, "
    "OOS queries often lie anomalously closer to the rival prototype than to the predicted prototype. "
    "A train-fitted ordered-pair calibration accounts for normal intent-specific classifier-geometry disagreements."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0112", "m_0189"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    if k < 2:
        g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
        d_tr = g.class_mahalanobis_sq(tr.feats)
        d_ev = g.class_mahalanobis_sq(ev.feats)
        p_tr = np.argmax(tr.logits, axis=1)
        p_ev = np.argmax(ev.logits, axis=1)
        raw_tr = d_tr[np.arange(len(p_tr)), p_tr]
        raw_ev = d_ev[np.arange(len(p_ev)), p_ev]
        center = np.mean(raw_tr)
        scale = np.std(raw_tr) + 1e-6
        return np.clip((raw_ev - center) / scale, -20.0, 20.0)

    g = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    d_tr = g.class_mahalanobis_sq(tr.feats)
    d_ev = g.class_mahalanobis_sq(ev.feats)

    p_tr = np.argmax(tr.logits, axis=1)
    p_ev = np.argmax(ev.logits, axis=1)

    masked_tr = tr.logits.copy()
    masked_tr[np.arange(len(p_tr)), p_tr] = -np.inf
    r_tr = np.argmax(masked_tr, axis=1)

    masked_ev = ev.logits.copy()
    masked_ev[np.arange(len(p_ev)), p_ev] = -np.inf
    r_ev = np.argmax(masked_ev, axis=1)

    raw_tr = d_tr[np.arange(len(p_tr)), p_tr] - d_tr[np.arange(len(p_tr)), r_tr]
    raw_ev = d_ev[np.arange(len(p_ev)), p_ev] - d_ev[np.arange(len(p_ev)), r_ev]

    global_mean = np.mean(raw_tr)
    global_second = np.mean(raw_tr * raw_tr)

    class_count = np.bincount(p_tr, minlength=k).astype(float)
    class_sum = np.bincount(p_tr, weights=raw_tr, minlength=k)
    class_second_sum = np.bincount(p_tr, weights=raw_tr * raw_tr, minlength=k)
    class_mean = np.divide(
        class_sum, class_count, out=np.full(k, global_mean), where=class_count > 0
    )
    class_second = np.divide(
        class_second_sum, class_count, out=np.full(k, global_second), where=class_count > 0
    )

    pair_id = p_tr * k + r_tr
    pair_count = np.bincount(pair_id, minlength=k * k).astype(float)
    pair_sum = np.bincount(pair_id, weights=raw_tr, minlength=k * k)
    pair_second_sum = np.bincount(pair_id, weights=raw_tr * raw_tr, minlength=k * k)

    pair_pred = np.repeat(np.arange(k), k)
    prior_mean = class_mean[pair_pred]
    prior_second = class_second[pair_pred]

    shrink = 8.0
    pair_mean = (pair_sum + shrink * prior_mean) / (pair_count + shrink)
    pair_second = (pair_second_sum + shrink * prior_second) / (pair_count + shrink)
    pair_var = np.maximum(pair_second - pair_mean * pair_mean, 1e-6)

    eval_pair = p_ev * k + r_ev
    z = (raw_ev - pair_mean[eval_pair]) / np.sqrt(pair_var[eval_pair])
    return np.clip(z, -20.0, 20.0)
