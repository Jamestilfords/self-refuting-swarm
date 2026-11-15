import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents tend to produce a characteristic class-conditional dispersion of centered logits. "
    "OOS queries often have unusually compressed relative logit evidence even when their absolute energy is not extreme."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_dispersion = np.sqrt(np.mean(tr_centered * tr_centered, axis=1))
    ev_dispersion = np.sqrt(np.mean(ev_centered * ev_centered, axis=1))

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    global_mean = float(np.mean(tr_dispersion))
    global_std = float(np.std(tr_dispersion))
    global_std = max(global_std, 1e-6)

    means = np.full(k, global_mean, dtype=float)
    stds = np.full(k, global_std, dtype=float)
    for c in range(k):
        values = tr_dispersion[tr_pred == c]
        if values.size >= 2:
            means[c] = float(np.mean(values))
            stds[c] = max(float(np.std(values)), 1e-6)

    score_values = (means[ev_pred] - ev_dispersion) / stds[ev_pred]
    return np.nan_to_num(score_values, nan=0.0, posinf=1e6, neginf=-1e6)
