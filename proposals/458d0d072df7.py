import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known-intent representations tend to have a class-conditional prototype isolation gap: "
    "their nearest intent prototype is meaningfully closer than the runner-up. OOS queries "
    "often occupy ambiguous regions between known intent clusters, producing an unusually small relative gap."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0176", "m_0221"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")
    d_tr = g.class_mahalanobis_sq(tr.pooled)
    d_ev = g.class_mahalanobis_sq(ev.pooled)

    def isolation_gap(d):
        two = np.partition(d, 1, axis=1)[:, :2]
        nearest = two[:, 0]
        second = two[:, 1]
        return (second - nearest) / (second + 1e-8)

    gap_tr = isolation_gap(d_tr)
    gap_ev = isolation_gap(d_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    n_classes = d_tr.shape[1]

    global_sorted = np.sort(gap_tr)
    out = np.empty(gap_ev.shape[0], dtype=float)

    for c in range(n_classes):
        ref = gap_tr[pred_tr == c]
        if ref.size < 8:
            ref = gap_tr
        ref = np.sort(ref)
        mask = pred_ev == c
        if np.any(mask):
            # Lower-than-typical prototype isolation is more OOS-like.
            ranks = np.searchsorted(ref, gap_ev[mask], side="right")
            out[mask] = 1.0 - ranks / (ref.size + 1.0)

    unseen = pred_ev >= n_classes
    if np.any(unseen):
        ranks = np.searchsorted(global_sorted, gap_ev[unseen], side="right")
        out[unseen] = 1.0 - ranks / (global_sorted.size + 1.0)

    return np.clip(out, 0.0, 1.0)
