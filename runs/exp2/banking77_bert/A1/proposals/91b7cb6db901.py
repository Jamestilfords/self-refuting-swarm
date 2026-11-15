import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "Softmax-weighted cosine support measures alignment with the intent prototypes favored by the classifier rather than only the single nearest prototype. OOS queries should have weak aggregate angular support among plausible in-scope intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    k = tr.logits.shape[1]

    prototypes = np.zeros((k, x_train.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(prototypes, tr.labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    similarities = x_eval @ prototypes.T
    probabilities = ctx.h.softmax(ev.logits)
    support = np.sum(probabilities * similarities, axis=1)
    return 1.0 - support
