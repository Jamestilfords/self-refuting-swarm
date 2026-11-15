import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Classifier uncertainty and classifier-selected feature-space mismatch are calibrated "
    "within predicted intents, then combined on a common empirical tail scale."
)
SOURCE = ""
PARENTS = ["msp", "d_full"]
MEMORY_USED = ["m_0009", "m_0015"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    train_all_d = g.class_mahalanobis_sq(tr.pooled)
    eval_all_d = g.class_mahalanobis_sq(ev.pooled)

    train_d = train_all_d[np.arange(len(tr.pooled)), train_pred]
    eval_d = eval_all_d[np.arange(len(ev.pooled)), eval_pred]

    train_u = ctx.h.msp(tr.logits)
    eval_u = ctx.h.msp(ev.logits)

    out_u = np.empty(len(ev.logits), dtype=float)
    out_d = np.empty(len(ev.logits), dtype=float)

    for c in range(tr.logits.shape[1]):
        mask = train_pred == c
        test_mask = eval_pred == c
        if not np.any(test_mask):
            continue

        if np.any(mask):
            ref_u = np.sort(train_u[mask])
            ref_d = np.sort(train_d[mask])
        else:
            ref_u = np.sort(train_u)
            ref_d = np.sort(train_d)

        out_u[test_mask] = (
            np.searchsorted(ref_u, eval_u[test_mask], side="right") / len(ref_u)
        )
        out_d[test_mask] = (
            np.searchsorted(ref_d, eval_d[test_mask], side="right") / len(ref_d)
        )

    return 0.5 * out_u + 0.5 * out_d
