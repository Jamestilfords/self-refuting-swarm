import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The centered logit norm measures the overall concentration of class evidence while "
    "discarding the shared logit offset that drives energy. OOS queries often have unusually "
    "diffuse evidence relative to examples assigned to the same intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    counts = np.bincount(tr.labels, minlength=k).astype(float)
    sums = np.bincount(tr.labels, weights=tr_norm, minlength=k)
    means = sums / np.maximum(counts, 1.0)

    residual = tr_norm - means[tr.labels]
    sq_sums = np.bincount(tr.labels, weights=residual * residual, minlength=k)
    stds = np.sqrt(sq_sums / np.maximum(counts, 1.0))

    global_std = np.std(tr_norm)
    stds = np.where(stds > 1e-6, stds, max(float(global_std), 1e-6))

    pred = np.argmax(ev.logits, axis=1)
    scores = (means[pred] - ev_norm) / stds[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
