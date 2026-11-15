import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Low maximum-softmax confidence and weak angular alignment with the classifier-predicted intent capture complementary uncertainty and semantic-support failures. Train-fitted empirical percentiles place both signals on a shared scale without using evaluation statistics."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0011", "m_0033"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    d = tr.pooled.shape[1]

    sums = np.zeros((k, d), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(sums, tr.labels, tr.pooled)
    prototypes = sums / np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    tr_x = tr.pooled / np.maximum(np.linalg.norm(tr.pooled, axis=1, keepdims=True), 1e-12)
    ev_x = ev.pooled / np.maximum(np.linalg.norm(ev.pooled, axis=1, keepdims=True), 1e-12)

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    tr_dist = 1.0 - np.sum(tr_x * prototypes[tr_pred], axis=1)
    ev_dist = 1.0 - np.sum(ev_x * prototypes[ev_pred], axis=1)

    tr_probs = ctx.h.softmax(tr.logits)
    ev_probs = ctx.h.softmax(ev.logits)
    tr_msp = 1.0 - np.max(tr_probs, axis=1)
    ev_msp = 1.0 - np.max(ev_probs, axis=1)

    def percentile(train_values, values):
        ref = np.sort(np.nan_to_num(train_values, nan=0.0, posinf=1e12, neginf=-1e12))
        val = np.nan_to_num(values, nan=0.0, posinf=1e12, neginf=-1e12)
        return np.searchsorted(ref, val, side="right") / float(max(ref.size, 1))

    p_msp = percentile(tr_msp, ev_msp)
    p_dist = percentile(tr_dist, ev_dist)
    return 0.5 * p_msp + 0.5 * p_dist
