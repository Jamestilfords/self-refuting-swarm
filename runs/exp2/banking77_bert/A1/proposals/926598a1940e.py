import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "OOS queries often have a smaller separation between their leading and runner-up intent logits than training examples assigned to the same predicted intent. Class-conditional train-fitted margin calibration preserves intent-specific confidence geometry beyond energy or maximum probability."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    tr_top2 = np.partition(tr.logits, k - 2, axis=1)[:, -2:]
    ev_top2 = np.partition(ev.logits, k - 2, axis=1)[:, -2:]
    tr_margin = tr_top2[:, 1] - tr_top2[:, 0]
    ev_margin = ev_top2[:, 1] - ev_top2[:, 0]

    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    counts = np.bincount(tr_pred, minlength=k).astype(float)
    total = max(float(tr_margin.size), 1.0)
    global_mean = float(np.mean(tr_margin))
    global_var = float(np.mean((tr_margin - global_mean) ** 2))
    global_scale = max(np.sqrt(global_var), 1e-6)

    sums = np.bincount(tr_pred, weights=tr_margin, minlength=k)
    raw_mean = sums / np.maximum(counts, 1.0)
    shrink = counts / (counts + 12.0)
    means = shrink * raw_mean + (1.0 - shrink) * global_mean

    centered = tr_margin - means[tr_pred]
    sq_sums = np.bincount(tr_pred, weights=centered ** 2, minlength=k)
    raw_var = sq_sums / np.maximum(counts, 1.0)
    scales = np.sqrt(
        shrink * raw_var + (1.0 - shrink) * (global_scale ** 2) + 1e-8
    )

    scores = (means[ev_pred] - ev_margin) / scales[ev_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
