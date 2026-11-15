import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents tend to exhibit a characteristic ranked logit-gap profile after the "
    "winning class is selected. OOS queries often have both unusually small winning margins "
    "and atypical relative spacing among the strongest competing intents."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    take = min(6, k)

    def signature(logits):
        top = np.sort(logits, axis=1)[:, -take:][:, ::-1]
        gaps = top[:, :-1] - top[:, 1:]
        span = np.maximum(top[:, :1] - top[:, -1:], 1e-8)
        profile = gaps / span
        margin = gaps[:, 0]
        return profile, margin

    tr_profile, tr_margin = signature(tr.logits)
    ev_profile, ev_margin = signature(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    dim = tr_profile.shape[1]
    global_mean = np.mean(tr_profile, axis=0)
    global_var = np.var(tr_profile, axis=0)
    global_var = np.maximum(global_var, 1e-6)

    global_margin_mean = float(np.mean(tr_margin))
    global_margin_std = float(np.std(tr_margin))
    global_margin_std = max(global_margin_std, 1e-6)

    means = np.tile(global_mean, (k, 1))
    variances = np.tile(global_var, (k, 1))
    margin_means = np.full(k, global_margin_mean)
    margin_stds = np.full(k, global_margin_std)

    for c in range(k):
        mask = tr_pred == c
        count = int(np.sum(mask))
        if count >= 5:
            p = tr_profile[mask]
            m = tr_margin[mask]
            means[c] = np.mean(p, axis=0)
            class_var = np.var(p, axis=0)
            shrink = count / (count + 20.0)
            variances[c] = np.maximum(
                shrink * class_var + (1.0 - shrink) * global_var,
                1e-6,
            )
            margin_means[c] = float(np.mean(m))
            margin_stds[c] = max(
                float(np.std(m)) * shrink + global_margin_std * (1.0 - shrink),
                1e-6,
            )

    delta = ev_profile - means[ev_pred]
    profile_score = np.sum((delta * delta) / variances[ev_pred], axis=1)

    margin_z = (margin_means[ev_pred] - ev_margin) / margin_stds[ev_pred]
    low_margin_score = np.maximum(margin_z, 0.0) ** 2

    return np.asarray(profile_score + low_margin_score, dtype=float)
