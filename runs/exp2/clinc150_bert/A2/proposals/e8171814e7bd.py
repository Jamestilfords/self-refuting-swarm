import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = "The normalized Dirichlet concentration vector captures the relative allocation of evidential mass across intents while discarding total concentration and vacuity. An OOS query can have a familiar predicted class but an evidence-allocation shape atypical of training queries assigned to that class."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0017"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.asarray(ctx.h.alpha(tr.logits), dtype=float)
    a_ev = np.asarray(ctx.h.alpha(ev.logits), dtype=float)
    a_tr = np.nan_to_num(a_tr, nan=1.0, posinf=1e6, neginf=1e-6)
    a_ev = np.nan_to_num(a_ev, nan=1.0, posinf=1e6, neginf=1e-6)
    a_tr = np.maximum(a_tr, 1e-8)
    a_ev = np.maximum(a_ev, 1e-8)

    z_tr = a_tr / np.maximum(a_tr.sum(axis=1, keepdims=True), 1e-8)
    z_ev = a_ev / np.maximum(a_ev.sum(axis=1, keepdims=True), 1e-8)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    counts = np.bincount(pred_tr, minlength=k).astype(float)
    sums = np.zeros((k, z_tr.shape[1]), dtype=float)
    sums2 = np.zeros_like(sums)
    np.add.at(sums, pred_tr, z_tr)
    np.add.at(sums2, pred_tr, z_tr * z_tr)

    global_mean = z_tr.mean(axis=0)
    global_var = np.maximum(z_tr.var(axis=0), 1e-8)

    denom = np.maximum(counts[:, None], 1.0)
    local_mean = sums / denom
    local_var = np.maximum(sums2 / denom - local_mean * local_mean, 0.0)

    weight = counts[:, None] / (counts[:, None] + 12.0)
    means = weight * local_mean + (1.0 - weight) * global_mean
    variances = weight * local_var + (1.0 - weight) * global_var
    variances = np.maximum(variances, 1e-8)

    diff = z_ev - means[pred_ev]
    scores = np.sum((diff * diff) / variances[pred_ev], axis=1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)

    if np.all(scores == scores[0]):
        scores = scores + 1e-12 * np.arange(scores.shape[0])

    return scores
