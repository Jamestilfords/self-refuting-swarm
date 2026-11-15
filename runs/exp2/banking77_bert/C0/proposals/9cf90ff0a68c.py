import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope examples tend to have both a class-specific logit direction and a characteristic centered-logit magnitude. OOS queries should deviate from the predicted class's learned logit shape and often have weaker centered-logit dispersion."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    z_tr = np.asarray(tr.logits, dtype=float)
    z_ev = np.asarray(ev.logits, dtype=float)
    y = np.asarray(tr.labels, dtype=int)

    k = z_tr.shape[1]
    eps = 1e-12

    tr_centered = z_tr - np.mean(z_tr, axis=1, keepdims=True)
    ev_centered = z_ev - np.mean(z_ev, axis=1, keepdims=True)

    tr_norm = np.sqrt(np.sum(tr_centered * tr_centered, axis=1))
    ev_norm = np.sqrt(np.sum(ev_centered * ev_centered, axis=1))

    tr_unit = tr_centered / np.maximum(tr_norm[:, None], eps)
    ev_unit = ev_centered / np.maximum(ev_norm[:, None], eps)

    counts = np.bincount(y, minlength=k).astype(float)
    proto = np.zeros((k, k), dtype=float)
    np.add.at(proto, y, tr_unit)
    proto /= np.maximum(counts[:, None], 1.0)
    proto_norm = np.sqrt(np.sum(proto * proto, axis=1, keepdims=True))
    proto /= np.maximum(proto_norm, eps)

    norm_sum = np.bincount(y, weights=tr_norm, minlength=k)
    norm_mean = norm_sum / np.maximum(counts, 1.0)
    norm_sq_sum = np.bincount(y, weights=tr_norm * tr_norm, minlength=k)
    norm_var = norm_sq_sum / np.maximum(counts, 1.0) - norm_mean * norm_mean
    norm_std = np.sqrt(np.maximum(norm_var, 1e-6))

    pred = np.argmax(z_ev, axis=1)
    cosine = np.sum(ev_unit * proto[pred], axis=1)
    angular_mismatch = 1.0 - np.clip(cosine, -1.0, 1.0)

    norm_deficit = (norm_mean[pred] - ev_norm) / norm_std[pred]
    return angular_mismatch + 0.25 * np.clip(norm_deficit, -6.0, 6.0)
