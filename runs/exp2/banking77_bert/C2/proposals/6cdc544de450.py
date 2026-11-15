import numpy as np
from sklearn.decomposition import PCA

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "A ViM-style residual is calibrated separately for the class predicted by the classifier, "
    "so a feature is anomalous when it falls outside the principal training subspace expected for its predicted intent."
)
SOURCE = ""
PARENTS = ["energy"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    n_components = min(96, tr.pooled.shape[1], max(1, tr.pooled.shape[0] - 1))

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=2718,
    )
    pca.fit(tr.pooled)

    tr_centered = tr.pooled - pca.mean_
    ev_centered = ev.pooled - pca.mean_
    tr_projection = pca.transform(tr.pooled) @ pca.components_
    ev_projection = pca.transform(ev.pooled) @ pca.components_

    tr_residual = np.linalg.norm(tr_centered - tr_projection, axis=1)
    ev_residual = np.linalg.norm(ev_centered - ev_projection, axis=1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    class_loc = np.empty(n_classes, dtype=float)
    class_scale = np.empty(n_classes, dtype=float)
    global_loc = float(np.median(tr_residual))
    global_scale = float(np.median(np.abs(tr_residual - global_loc))) * 1.4826
    global_scale = max(global_scale, 1e-6)

    for c in range(n_classes):
        values = tr_residual[train_pred == c]
        if values.size >= 8:
            loc = float(np.median(values))
            scale = float(np.median(np.abs(values - loc))) * 1.4826
            class_loc[c] = loc
            class_scale[c] = max(scale, 0.25 * global_scale, 1e-6)
        else:
            class_loc[c] = global_loc
            class_scale[c] = global_scale

    residual_z = (ev_residual - class_loc[eval_pred]) / class_scale[eval_pred]

    train_energy = ctx.h.energy(tr.logits)
    energy_loc = float(np.median(train_energy))
    energy_scale = float(np.median(np.abs(train_energy - energy_loc))) * 1.4826
    energy_scale = max(energy_scale, 1e-6)
    energy_z = (ctx.h.energy(ev.logits) - energy_loc) / energy_scale

    return residual_z + 0.35 * energy_z
