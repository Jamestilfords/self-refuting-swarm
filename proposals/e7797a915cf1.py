import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Rather than relying only on the nearest angular prototype, this score measures aggregate cosine support "
    "from the full set of known intent directions. OOS queries should have weak soft-nearest prototype support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0009", "m_0020", "m_0075"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels
    n_classes = tr.logits.shape[1]

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    similarities = x_eval @ prototypes.T
    temperature = 0.12
    scaled = similarities / temperature
    row_max = np.max(scaled, axis=1, keepdims=True)
    log_support = row_max[:, 0] + np.log(
        np.sum(np.exp(scaled - row_max), axis=1)
    )

    return np.nan_to_num(-temperature * log_support, nan=0.0, posinf=1e6, neginf=-1e6)
