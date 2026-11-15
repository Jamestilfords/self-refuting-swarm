import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL concentration vectors encode more than vacuity: their normalized class-evidence "
    "shape should resemble the characteristic evidence profile of the predicted in-scope intent. "
    "A Jensen-Shannon divergence from train-fitted class-conditional Dirichlet shapes flags atypical evidence allocation."
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
    sums = np.zeros((n_classes, p_tr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(sums, tr.labels, p_tr)

    global_shape = np.mean(p_tr, axis=0, keepdims=True)
    prototypes = (sums + global_shape) / (counts[:, None] + 1.0)
    prototypes /= np.sum(prototypes, axis=1, keepdims=True)

    pred = np.argmax(a_ev, axis=1)
    q = np.maximum(prototypes[pred], eps)
    m = 0.5 * (p_ev + q)

    kl_p = np.sum(p_ev * (np.log(p_ev) - np.log(m)), axis=1)
    kl_q = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * (kl_p + kl_q)
