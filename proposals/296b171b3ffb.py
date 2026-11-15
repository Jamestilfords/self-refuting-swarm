import numpy as np

FAMILY = "knn"
ARM = "ce"
RATIONALE = (
    "A query selected as an intent should have a locally dense neighborhood among training "
    "examples of that same intent. Comparing its predicted-class neighbor radius to both a "
    "train-calibrated class radius and the unrestricted neighbor radius detects unsupported "
    "or label-incoherent local neighborhoods."
)
SOURCE = "https://proceedings.mlr.press/v162/sun22d/sun22d.pdf"
PARENTS = []
MEMORY_USED = ["m_0055", "m_0016"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = tr.pooled
    x_eval = ev.pooled
    labels = np.asarray(tr.labels)
    pred = np.argmax(ev.logits, axis=1)
    n_train = len(labels)

    global_k = min(10, n_train)
    global_eval = ctx.h.knn(x_train, x_eval, k=global_k, l2=True)
    global_train_k = min(global_k + 1, n_train)
    global_scale = np.median(
        ctx.h.knn(x_train, x_train, k=global_train_k, l2=True)
    )
    global_scale = max(float(global_scale), 1e-8)

    out = np.empty(len(x_eval), dtype=float)
    for c in np.unique(pred):
        mask_eval = pred == c
        mask_train = labels == c
        xc = x_train[mask_train]
        nc = len(xc)

        if nc == 0:
            class_radius = global_eval[mask_eval]
            class_scale = global_scale
        else:
            k = min(10, nc)
            class_radius = ctx.h.knn(xc, x_eval[mask_eval], k=k, l2=True)
            if nc > 1:
                train_k = min(k + 1, nc)
                class_scale = np.median(ctx.h.knn(xc, xc, k=train_k, l2=True))
                class_scale = max(float(class_scale), 1e-8)
            else:
                class_scale = global_scale

        local_surprise = np.log1p(class_radius / class_scale)
        label_incoherence = np.log1p(class_radius / (global_eval[mask_eval] + 1e-8))
        out[mask_eval] = local_surprise + 0.5 * label_incoherence

    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
