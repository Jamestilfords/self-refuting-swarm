import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "EDL in-scope examples should produce class-specific relative Dirichlet concentration patterns, even after removing total evidence magnitude. Queries whose normalized log-concentration shape is unlike every training intent receive a high score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    alpha_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    # Remove the common log-concentration offset so the score measures
    # Dirichlet shape rather than total evidence/vacuity.
    shape_tr = np.log(alpha_tr)
    shape_tr = shape_tr - shape_tr.mean(axis=1, keepdims=True)
    shape_ev = np.log(alpha_ev)
    shape_ev = shape_ev - shape_ev.mean(axis=1, keepdims=True)

    scale = shape_tr.std(axis=0)
    scale = np.maximum(scale, 1e-4)
    z_tr = shape_tr / scale
    z_ev = shape_ev / scale

    labels = tr.labels.astype(int)
    classes = np.unique(labels)
    prototypes = np.empty((classes.size, z_tr.shape[1]), dtype=float)
    for i, cls in enumerate(classes):
        prototypes[i] = z_tr[labels == cls].mean(axis=0)

    ev_norm = np.sum(z_ev * z_ev, axis=1, keepdims=True)
    proto_norm = np.sum(prototypes * prototypes, axis=1)[None, :]
    distances = ev_norm + proto_norm - 2.0 * z_ev.dot(prototypes.T)
    return np.maximum(np.min(distances, axis=1), 0.0)
