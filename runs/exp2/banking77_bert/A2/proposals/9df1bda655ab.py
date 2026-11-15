import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "The relative shape of Dirichlet concentrations encodes how evidence is allocated across classes, independently of vacuity or total concentration. OOS inputs should align poorly with the class-specific concentration-shape prototypes learned from in-scope training examples."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    def shape(alpha):
        z = np.log(alpha)
        z = z - z.mean(axis=1, keepdims=True)
        n = np.sqrt(np.sum(z * z, axis=1, keepdims=True))
        return z / np.maximum(n, 1e-12)

    z_tr = shape(a_tr)
    z_ev = shape(a_ev)

    n_classes = tr.logits.shape[1]
    sums = np.zeros((n_classes, z_tr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=n_classes).astype(float)
    np.add.at(sums, tr.labels, z_tr)
    proto = sums / np.maximum(counts[:, None], 1.0)

    proto_norm = np.sqrt(np.sum(proto * proto, axis=1, keepdims=True))
    proto = proto / np.maximum(proto_norm, 1e-12)

    similarity = z_ev @ proto.T
    return 1.0 - np.max(similarity, axis=1)
