import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two margin normalized by centered logit norm measures angular confidence rather than raw logit scale. OOS queries should have a smaller fraction of their logit energy concentrated in the winning-class direction."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0011"]


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    tr_centered = tr_logits - tr_logits.mean(axis=1, keepdims=True)
    ev_centered = ev_logits - ev_logits.mean(axis=1, keepdims=True)

    tr_top2 = np.partition(tr_centered, -2, axis=1)[:, -2:]
    ev_top2 = np.partition(ev_centered, -2, axis=1)[:, -2:]

    tr_margin = tr_top2[:, 1] - tr_top2[:, 0]
    ev_margin = ev_top2[:, 1] - ev_top2[:, 0]

    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    scale = np.median(tr_norm)
    eps = max(float(scale) * 1e-8, 1e-12)

    tr_ratio = tr_margin / (tr_norm + eps)
    ev_ratio = ev_margin / (ev_norm + eps)

    center = np.median(tr_ratio)
    mad = np.median(np.abs(tr_ratio - center))
    denom = max(float(1.4826 * mad), 1e-8)

    out = (center - ev_ratio) / denom
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
