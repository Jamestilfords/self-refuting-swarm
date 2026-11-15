import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL in-scope examples should exhibit a learned agreement between normalized Dirichlet evidence allocation "
    "and discriminative softmax allocation. Class-conditional surprise in this allocation-agreement shape flags "
    "unfamiliar evidential competition without being a monotone function of energy or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def descriptors(logits, alpha):
        z = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        z = z - np.max(z, axis=1, keepdims=True)
        q = np.exp(np.clip(z, -60.0, 0.0))
        q = q / np.maximum(np.sum(q, axis=1, keepdims=True), eps)

        a = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=eps)
        a = np.maximum(a, eps)
        p = a / np.maximum(np.sum(a, axis=1, keepdims=True), eps)

        m = 0.5 * (p + q)
        js = 0.5 * np.sum(
            p * (np.log(np.maximum(p, eps)) - np.log(np.maximum(m, eps))),
            axis=1,
        ) + 0.5 * np.sum(
            q * (np.log(np.maximum(q, eps)) - np.log(np.maximum(m, eps))),
            axis=1,
        )

        lp = np.log(np.maximum(p, eps))
        lq = np.log(np.maximum(q, eps))
        lp = lp - np.mean(lp, axis=1, keepdims=True)
        lq = lq - np.mean(lq, axis=1, keepdims=True)
        denom = np.linalg.norm(lp, axis=1) * np.linalg.norm(lq, axis=1)
        cosine_disagreement = 1.0 - np.sum(lp * lq, axis=1) / np.maximum(denom, eps)

        return np.column_stack((js, cosine_disagreement))

    x_train = descriptors(tr.logits, ctx.h.alpha(tr.logits))
    x_eval = descriptors(ev.logits, ctx.h.alpha(ev.logits))

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    global_mean = np.mean(x_train, axis=0)
    if x_train.shape[0] > 1:
        global_cov = np.cov(x_train, rowvar=False)
    else:
        global_cov = np.eye(2)

    global_cov = np.atleast_2d(global_cov)
    scale = max(float(np.trace(global_cov) / 2.0), 1e-6)
    global_cov = global_cov + (1e-5 * scale) * np.eye(2)

    means = np.empty((n_classes, 2), dtype=float)
    inv_covs = np.empty((n_classes, 2, 2), dtype=float)

    for c in range(n_classes):
        xc = x_train[train_pred == c]
        if xc.shape[0] == 0:
            mean = global_mean
            cov = global_cov
        else:
            mean = np.mean(xc, axis=0)
            if xc.shape[0] >= 3:
                local_cov = np.atleast_2d(np.cov(xc, rowvar=False))
                shrink = min(0.8, 4.0 / float(xc.shape[0] + 4))
                cov = (1.0 - shrink) * local_cov + shrink * global_cov
            else:
                cov = global_cov
            cov = cov + (1e-5 * scale) * np.eye(2)

        means[c] = mean
        inv_covs[c] = np.linalg.pinv(cov)

    delta = x_eval - means[eval_pred]
    chosen_inv = inv_covs[eval_pred]
    scores = np.einsum("ni,nij,nj->n", delta, chosen_inv, delta)
    return np.log1p(np.maximum(scores, 0.0))
