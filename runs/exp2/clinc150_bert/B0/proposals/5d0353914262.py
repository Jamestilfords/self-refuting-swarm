import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The relative allocation of Dirichlet evidence across intents has class-specific in-scope patterns that are not captured by total evidence or vacuity. OOS queries can have an atypical compositional evidence shape even when their overall concentration is similar to in-scope queries."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0097", "m_0117"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    def clr(alpha):
        log_a = np.log(alpha)
        return log_a - log_a.mean(axis=1, keepdims=True)

    z_tr = clr(a_tr)
    z_ev = clr(a_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = z_tr.mean(axis=0)
    global_var = z_tr.var(axis=0) + 1e-6

    counts = np.bincount(pred_tr, minlength=k).astype(float)
    sums = np.zeros((k, z_tr.shape[1]), dtype=float)
    sums2 = np.zeros_like(sums)
    np.add.at(sums, pred_tr, z_tr)
    np.add.at(sums2, pred_tr, z_tr * z_tr)

    raw_mean = sums / np.maximum(counts[:, None], 1.0)
    raw_var = sums2 / np.maximum(counts[:, None], 1.0) - raw_mean * raw_mean
    raw_var = np.maximum(raw_var, 1e-6)

    strength = 12.0
    w = counts[:, None] / (counts[:, None] + strength)
    means = w * raw_mean + (1.0 - w) * global_mean
    variances = w * raw_var + (1.0 - w) * global_var
    variances = np.maximum(variances, 1e-5)

    delta = z_ev - means[pred_ev]
    scores = np.mean((delta * delta) / variances[pred_ev], axis=1)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
