import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidential mass is allocated across intents, independently of total evidence or vacuity. "
    "In-scope examples should resemble the characteristic evidence-allocation profile of their predicted intent, while OOS examples often have atypical concentration shapes."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)
    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    n_classes = tr.logits.shape[1]
    global_profile = np.mean(p_tr, axis=0)
    prototypes = np.tile(global_profile, (n_classes, 1))
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)

    sums = np.zeros((n_classes, p_tr.shape[1]), dtype=float)
    np.add.at(sums, tr.labels, p_tr)
    present = counts > 0
    prototypes[present] = sums[present] / counts[present, None]

    pred = np.argmax(ev.logits, axis=1)
    q = np.maximum(prototypes[pred], eps)
    m = 0.5 * (p_ev + q)

    js = 0.5 * np.sum(
        p_ev * np.log(np.maximum(p_ev, eps) / m)
        + q * np.log(q / m),
        axis=1,
    )
    return np.nan_to_num(js, nan=0.0, posinf=1e6, neginf=0.0)
