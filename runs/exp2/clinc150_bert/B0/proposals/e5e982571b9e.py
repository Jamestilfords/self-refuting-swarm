import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "Evidential in-scope examples should exhibit class-specific Dirichlet concentration and "
    "probability-dispersion patterns, whereas OOS inputs can have atypical total evidence or "
    "an atypical allocation of evidence across intents even when their maximum class is high."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0058", "m_0063"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)
    k = a_tr.shape[1]

    def evidence_shape(alpha):
        total = np.sum(alpha, axis=1)
        probs = alpha / total[:, None]
        entropy = -np.sum(probs * np.log(probs), axis=1) / np.log(max(k, 2))
        concentration = np.log(total)
        return np.column_stack((concentration, entropy))

    x_tr = evidence_shape(a_tr)
    x_ev = evidence_shape(a_ev)

    med = np.zeros((k, 2), dtype=float)
    scale = np.ones((k, 2), dtype=float)
    global_med = np.median(x_tr, axis=0)
    global_mad = np.median(np.abs(x_tr - global_med), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 1e-4)

    for c in range(k):
        xc = x_tr[tr.labels == c]
        if len(xc) >= 3:
            mc = np.median(xc, axis=0)
            mad = np.median(np.abs(xc - mc), axis=0)
            med[c] = mc
            scale[c] = np.maximum(1.4826 * mad, 0.15 * global_scale)
        else:
            med[c] = global_med
            scale[c] = global_scale

    pred = np.argmax(a_ev, axis=1)
    z = (x_ev - med[pred]) / scale[pred]
    return np.sum(np.minimum(z * z, 100.0), axis=1)
