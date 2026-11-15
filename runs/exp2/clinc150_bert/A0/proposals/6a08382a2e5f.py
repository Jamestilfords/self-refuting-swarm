import numpy as np
from sklearn.linear_model import LogisticRegression

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder density deficits are fused with weights learned from train-only pseudo-OOS boundary samples formed between neighboring intent classes. This calibrates each representation level by how strongly it distinguishes supported in-scope examples from cross-intent feature regions."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = ["m_0001"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied", whiten=True)
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied", whiten=True)

    classes, inverse = np.unique(tr.labels, return_inverse=True)
    n_classes = len(classes)

    port_centers = np.zeros((n_classes, tr.feats.shape[1]), dtype=float)
    full_centers = np.zeros((n_classes, tr.pooled.shape[1]), dtype=float)
    for i in range(n_classes):
        mask = inverse == i
        port_centers[i] = np.mean(tr.feats[mask], axis=0)
        full_centers[i] = np.mean(tr.pooled[mask], axis=0)

    center_dist = (
        np.sum(full_centers * full_centers, axis=1)[:, None]
        + np.sum(full_centers * full_centers, axis=1)[None, :]
        - 2.0 * full_centers.dot(full_centers.T)
    )
    center_dist += np.eye(n_classes) * np.inf
    neighbor = np.argmin(center_dist, axis=1)

    synth_port = 0.5 * tr.feats + 0.5 * port_centers[neighbor[inverse]]
    synth_full = 0.5 * tr.pooled + 0.5 * full_centers[neighbor[inverse]]

    real_features = np.column_stack((
        -g_port.log_density(tr.feats),
        -g_full.log_density(tr.pooled),
    ))
    synth_features = np.column_stack((
        -g_port.log_density(synth_port),
        -g_full.log_density(synth_full),
    ))

    fit_x = np.vstack((real_features, synth_features))
    fit_y = np.concatenate((
        np.zeros(len(real_features), dtype=int),
        np.ones(len(synth_features), dtype=int),
    ))

    mean = np.mean(fit_x, axis=0)
    scale = np.std(fit_x, axis=0)
    scale = np.maximum(scale, 1e-8)
    fit_x = np.nan_to_num((fit_x - mean) / scale, nan=0.0, posinf=20.0, neginf=-20.0)

    clf = LogisticRegression(
        C=0.5,
        max_iter=200,
        solver="lbfgs",
        random_state=17,
    )
    clf.fit(fit_x, fit_y)

    eval_features = np.column_stack((
        -g_port.log_density(ev.feats),
        -g_full.log_density(ev.pooled),
    ))
    eval_features = np.nan_to_num(
        (eval_features - mean) / scale,
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )

    return clf.decision_function(eval_features)
