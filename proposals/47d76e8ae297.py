import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidence is allocated among intents independently of total evidence. "
    "A predicted-intent-specific two-sided tail score for concentration (Simpson) detects unusually diffuse or unusually concentrated evidential competition."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0025", "m_0050", "m_0129"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    # Simpson concentration measures allocation shape, not total evidence.
    s_tr = np.sum(p_tr * p_tr, axis=1)
    s_ev = np.sum(p_ev * p_ev, axis=1)

    y_tr = np.argmax(tr.logits, axis=1)
    y_ev = np.argmax(ev.logits, axis=1)
    global_sorted = np.sort(s_tr)

    out = np.empty(len(s_ev), dtype=float)
    for cls in np.unique(y_ev):
        mask_ev = y_ev == cls
        reference = np.sort(s_tr[y_tr == cls])
        if reference.size < 8:
            reference = global_sorted

        left = np.searchsorted(reference, s_ev[mask_ev], side="left")
        right = np.searchsorted(reference, s_ev[mask_ev], side="right")
        n = float(reference.size)

        lower = (left + 1.0) / (n + 2.0)
        upper = (n - right + 1.0) / (n + 2.0)
        tail = np.minimum(lower, upper)

        out[mask_ev] = -np.log(np.maximum(tail, 1e-12)) + 1e-6 * s_ev[mask_ev]

    return out
