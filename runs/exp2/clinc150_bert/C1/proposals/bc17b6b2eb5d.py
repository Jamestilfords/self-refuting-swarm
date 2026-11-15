import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A class-conditional residual score measures distance outside the local principal subspace of the class predicted by the classifier. OOS queries that are assigned an in-scope label but do not share that label's low-dimensional feature structure should have unusually large residuals."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_train = np.asarray(tr.feats, dtype=np.float64)
    x_eval = np.asarray(ev.feats, dtype=np.float64)
    labels = np.asarray(tr.labels, dtype=np.int64)
    pred = np.argmax(np.asarray(ev.logits), axis=1)

    n_classes = int(max(np.max(labels), np.max(pred)) + 1)
    dim = x_train.shape[1]
    scores = np.empty(x_eval.shape[0], dtype=np.float64)

    global_mean = np.mean(x_train, axis=0)
    global_centered = x_train - global_mean
    global_cov = (global_centered.T @ global_centered) / max(len(x_train) - 1, 1)
    global_vals, global_vecs = np.linalg.eigh(global_cov)
    global_rank = min(32, dim)
    global_basis = global_vecs[:, -global_rank:]

    for c in range(n_classes):
        eval_idx = np.flatnonzero(pred == c)
        if eval_idx.size == 0:
            continue

        train_idx = np.flatnonzero(labels == c)
        if train_idx.size >= 4:
            class_x = x_train[train_idx]
            mean = np.mean(class_x, axis=0)
            centered = class_x - mean
            cov = (centered.T @ centered) / max(class_x.shape[0] - 1, 1)
            vals, vecs = np.linalg.eigh(cov)

            rank = min(16, dim, max(class_x.shape[0] - 2, 1))
            basis = vecs[:, -rank:]

            train_proj = centered @ basis
            train_resid = np.sqrt(np.maximum(
                np.sum(centered * centered, axis=1) - np.sum(train_proj * train_proj, axis=1),
                0.0,
            ))
        else:
            mean = global_mean
            basis = global_basis
            train_proj = global_centered @ basis
            train_resid = np.sqrt(np.maximum(
                np.sum(global_centered * global_centered, axis=1)
                - np.sum(train_proj * train_proj, axis=1),
                0.0,
            ))

        median = float(np.median(train_resid))
        mad = float(np.median(np.abs(train_resid - median)))
        scale = max(1.4826 * mad, 1e-6)

        centered_eval = x_eval[eval_idx] - mean
        eval_proj = centered_eval @ basis
        eval_resid = np.sqrt(np.maximum(
            np.sum(centered_eval * centered_eval, axis=1)
            - np.sum(eval_proj * eval_proj, axis=1),
            0.0,
        ))
        scores[eval_idx] = (eval_resid - median) / scale

    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
