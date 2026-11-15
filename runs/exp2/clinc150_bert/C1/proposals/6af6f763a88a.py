import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "In-scope examples should exhibit class-specific Dirichlet concentration shapes, whereas OOS queries often produce atypical relative concentration patterns even when total evidence is similar. This score measures angular mismatch between an item's centered log-concentration profile and the profile learned for its predicted class."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-8)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-8)

    z_tr = np.log(a_tr)
    z_ev = np.log(a_ev)
    z_tr = z_tr - z_tr.mean(axis=1, keepdims=True)
    z_ev = z_ev - z_ev.mean(axis=1, keepdims=True)

    z_tr /= np.maximum(np.linalg.norm(z_tr, axis=1, keepdims=True), 1e-8)
    z_ev /= np.maximum(np.linalg.norm(z_ev, axis=1, keepdims=True), 1e-8)

    k = tr.logits.shape[1]
    centers = np.zeros((k, z_tr.shape[1]), dtype=float)
    counts = np.bincount(tr.labels, minlength=k).astype(float)
    np.add.at(centers, tr.labels, z_tr)
    centers /= np.maximum(counts[:, None], 1.0)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)

    pred = np.argmax(a_ev, axis=1)
    similarity = np.sum(z_ev * centers[pred], axis=1)

    strength = np.sum(a_ev, axis=1)
    tr_strength = np.sum(a_tr, axis=1)
    log_strength_center = np.median(np.log1p(tr_strength))
    strength_departure = np.abs(np.log1p(strength) - log_strength_center)

    return 1.0 - similarity + 0.05 * strength_departure
