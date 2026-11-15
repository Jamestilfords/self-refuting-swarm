import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions capture subtle representation deviations "
    "that dominant semantic directions can obscure. A predicted-intent-conditioned training "
    "tail calibration flags samples unusually displaced in those low-variance directions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0031"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    # The partial model is fitted only to labeled in-scope training representations.
    part = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_raw = np.asarray(part.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_raw = np.asarray(part.bottom(ev.pooled), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    # Per predicted-class empirical upper-tail probabilities, fitted exclusively on train.
    sorted_by_class = []
    global_sorted = np.sort(train_raw[np.isfinite(train_raw)])
    if global_sorted.size == 0:
        global_sorted = np.array([0.0])

    for c in range(k):
        values = train_raw[(train_pred == c) & np.isfinite(train_raw)]
        if values.size < 8:
            values = global_sorted
        else:
            values = np.sort(values)
        sorted_by_class.append(values)

    scores = np.empty(eval_raw.shape[0], dtype=float)
    for c in range(k):
        mask = eval_pred == c
        if not np.any(mask):
            continue
        ref = sorted_by_class[c]
        vals = np.nan_to_num(
            eval_raw[mask],
            nan=float(ref[-1]),
            posinf=float(ref[-1]),
            neginf=float(ref[0]),
        )
        rank = np.searchsorted(ref, vals, side="right")
        survival = (ref.size - rank + 0.5) / (ref.size + 1.0)
        scores[mask] = -np.log(np.maximum(survival, 1e-12))

    return scores
