import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries are often assigned a known intent with a smaller gap between the two most competitive logits. "
    "Predicted-class robust calibration accounts for intents with intrinsically different logit-margin scales."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def top_two_margin(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    tr_margin = top_two_margin(tr.logits)
    ev_margin = top_two_margin(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)

    global_med = np.median(tr_margin)
    global_q1, global_q3 = np.quantile(tr_margin, [0.25, 0.75])
    global_scale = max(global_q3 - global_q1, 1e-6)

    med = np.full(ev_margin.shape, global_med, dtype=float)
    scale = np.full(ev_margin.shape, global_scale, dtype=float)

    for c in range(tr.logits.shape[1]):
        values = tr_margin[tr_pred == c]
        if values.size >= 4:
            q1, q2, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            mask = ev_pred == c
            med[mask] = q2
            scale[mask] = max(q3 - q1, 1e-6)

    z = np.clip((med - ev_margin) / scale, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))
