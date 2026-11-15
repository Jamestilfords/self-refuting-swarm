import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Maximum-softmax uncertainty and pooled-feature prototype distance capture complementary "
    "failure modes. Per-predicted-intent empirical tail calibration makes both signals comparable "
    "while preserving intent-specific in-scope variability."
)
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0037", "m_0047"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_msp = ctx.h.msp(tr.logits)
    eval_msp = ctx.h.msp(ev.logits)
    train_dist = g.class_mahalanobis_sq(tr.pooled)[
        np.arange(len(tr.logits)), train_pred
    ]
    eval_dist = g.class_mahalanobis_sq(ev.pooled)[
        np.arange(len(ev.logits)), eval_pred
    ]

    out_msp = np.empty(len(ev.logits), dtype=float)
    out_dist = np.empty(len(ev.logits), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        idx = eval_pred == c
        if not np.any(idx):
            continue

        if np.sum(mask) < 8:
            ref_msp = np.sort(train_msp)
            ref_dist = np.sort(train_dist)
        else:
            ref_msp = np.sort(train_msp[mask])
            ref_dist = np.sort(train_dist[mask])

        out_msp[idx] = np.searchsorted(ref_msp, eval_msp[idx], side="right") / (
            len(ref_msp) + 1.0
        )
        out_dist[idx] = np.searchsorted(ref_dist, eval_dist[idx], side="right") / (
            len(ref_dist) + 1.0
        )

    score_value = 0.5 * (out_msp + out_dist) + 0.5 * np.maximum(out_msp, out_dist)
    return np.clip(score_value, 0.0, 1.0)
