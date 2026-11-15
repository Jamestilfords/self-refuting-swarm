import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL out-of-scope inputs should often lack the class-specific evidence dominance seen for "
    "in-scope examples. A train-calibrated top-versus-runner-up evidence margin detects atypical "
    "Dirichlet concentration shape rather than total evidence or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0057"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.asarray(ctx.h.alpha(tr.logits), dtype=float)
    a_ev = np.asarray(ctx.h.alpha(ev.logits), dtype=float)

    e_tr = np.maximum(a_tr - 1.0, 0.0)
    e_ev = np.maximum(a_ev - 1.0, 0.0)

    def dominance(evidence):
        top2 = np.partition(evidence, -2, axis=1)[:, -2:]
        top = top2[:, 1]
        second = top2[:, 0]
        total = np.sum(evidence, axis=1)
        return (top - second) / (total + 1e-8)

    d_tr = dominance(e_tr)
    d_ev = dominance(e_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = np.mean(d_tr)
    global_std = np.std(d_tr) + 1e-6
    means = np.full(k, global_mean)
    stds = np.full(k, global_std)

    for c in range(k):
        vals = d_tr[pred_tr == c]
        if vals.size >= 2:
            means[c] = np.mean(vals)
            stds[c] = np.std(vals) + 1e-6

    scores = (means[pred_ev] - d_ev) / stds[pred_ev]
    tie_break = np.asarray(ev.logits, dtype=float) @ np.linspace(-1.0, 1.0, k)
    return np.nan_to_num(scores + 1e-10 * tie_break, nan=0.0, posinf=1e6, neginf=-1e6)
