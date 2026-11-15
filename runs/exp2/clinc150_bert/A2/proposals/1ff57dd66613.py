import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = (
    "A classifier-selected intent may be close to its centroid while still lying outside that "
    "intent's local principal subspace. Class-conditional residual energy detects this off-manifold "
    "novelty and is calibrated against residuals of training examples from the same predicted intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0003", "m_0007", "m_0009"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.pooled, dtype=np.float64)
    x_eval = np.asarray(ev.pooled, dtype=np.float64)
    y_train = np.asarray(tr.labels)
    pred_eval = np.argmax(ev.logits, axis=1)

    global_mean = x_train.mean(axis=0)
    global_centered = x_train - global_mean
    _, _, global_vt = np.linalg.svd(global_centered, full_matrices=False)
    global_rank = min(64, global_vt.shape[0])
    global_basis = global_vt[:global_rank]

    def residual_norm(x, mean, basis):
        z = x - mean
        projection = (z @ basis.T) @ basis
        return np.sqrt(np.maximum(np.sum((z - projection) ** 2, axis=1), 0.0))

    global_train_resid = residual_norm(x_train, global_mean, global_basis)
    global_loc = np.median(global_train_resid)
    global_scale = np.median(np.abs(global_train_resid - global_loc)) * 1.4826
    global_scale = max(global_scale, 1e-8)

    scores = np.empty(x_eval.shape[0], dtype=np.float64)
    classes = np.unique(pred_eval)

    for cls in classes:
        eval_mask = pred_eval == cls
        train_mask = y_train == cls
        class_x = x_train[train_mask]

        if class_x.shape[0] < 8:
            resid = residual_norm(x_eval[eval_mask], global_mean, global_basis)
            scores[eval_mask] = (resid - global_loc) / global_scale
            continue

        mean = class_x.mean(axis=0)
        centered = class_x - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        energy = singular ** 2
        cumulative = np.cumsum(energy)
        if cumulative[-1] > 1e-12:
            rank = int(np.searchsorted(cumulative, 0.90 * cumulative[-1]) + 1)
        else:
            rank = 1
        rank = min(max(rank, 1), 64, vt.shape[0])
        basis = vt[:rank]

        train_resid = residual_norm(class_x, mean, basis)
        loc = np.median(train_resid)
        scale = np.median(np.abs(train_resid - loc)) * 1.4826
        if scale < 1e-8:
            scale = global_scale

        eval_resid = residual_norm(x_eval[eval_mask], mean, basis)
        scores[eval_mask] = (eval_resid - loc) / scale

    energy = ctx.h.energy(ev.logits)
    train_energy = ctx.h.energy(tr.logits)
    e_loc = np.median(train_energy)
    e_scale = max(np.median(np.abs(train_energy - e_loc)) * 1.4826, 1e-8)
    return scores + 0.03 * (energy - e_loc) / e_scale
