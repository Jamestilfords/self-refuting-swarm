import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "EDL concentration vectors encode more than vacuity: their class-specific relative evidence "
    "patterns and total evidence should be typical for supported in-scope intents. A diagonal, "
    "train-fitted residual on centered log-concentrations detects atypical evidential shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0126", "m_0143"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Centering log concentrations removes their common evidence-scale component,
    # retaining the Dirichlet shape independently of vacuity.
    z_tr = np.log(a_tr)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = np.log(a_ev)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    # Include evidence magnitude separately, but calibrate it conditionally on intent.
    e_tr = np.log(a_tr.sum(axis=1, keepdims=True))
    e_ev = np.log(a_ev.sum(axis=1, keepdims=True))
    x_tr = np.concatenate([z_tr, e_tr], axis=1)
    x_ev = np.concatenate([z_ev, e_ev], axis=1)

    labels = tr.labels.astype(int)
    pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]
    d = x_tr.shape[1]

    global_mean = x_tr.mean(axis=0)
    global_var = x_tr.var(axis=0) + 1e-5
    means = np.empty((n_classes, d), dtype=float)
    vars_ = np.empty((n_classes, d), dtype=float)

    for c in range(n_classes):
        xc = x_tr[labels == c]
        count = xc.shape[0]
        if count == 0:
            means[c] = global_mean
            vars_[c] = global_var
        else:
            mean_c = xc.mean(axis=0)
            var_c = xc.var(axis=0)
            # Shrink sparse intent estimates toward pooled evidential statistics.
            weight = count / (count + 20.0)
            means[c] = weight * mean_c + (1.0 - weight) * global_mean
            vars_[c] = weight * var_c + (1.0 - weight) * global_var + 1e-5

    diff = x_ev - means[pred]
    score = np.mean((diff * diff) / vars_[pred], axis=1)
    return np.nan_to_num(score, nan=0.0, posinf=1e12, neginf=0.0)
