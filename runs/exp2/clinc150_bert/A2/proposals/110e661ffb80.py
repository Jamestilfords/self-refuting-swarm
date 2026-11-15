import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "After removing the common logit offset captured by energy, in-scope intents tend to have "
    "class-characteristic relative-logit norms. Atypical centered-logit concentration for the predicted intent signals OOS risk."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    tr_centered = tr.logits - np.mean(tr.logits, axis=1, keepdims=True)
    ev_centered = ev.logits - np.mean(ev.logits, axis=1, keepdims=True)
    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    k = tr.logits.shape[1]
    global_med = np.median(tr_norm)
    global_mad = np.median(np.abs(tr_norm - global_med)) + 1e-6

    med = np.full(k, global_med, dtype=float)
    mad = np.full(k, global_mad, dtype=float)
    for c in range(k):
        values = tr_norm[tr.labels == c]
        if values.size:
            med[c] = np.median(values)
            mad[c] = np.median(np.abs(values - med[c])) + 1e-6

    pred = np.argmax(ev.logits, axis=1)
    scores = np.abs(ev_norm - med[pred]) / mad[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
