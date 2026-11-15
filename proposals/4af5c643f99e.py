import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "After removing the class-mean discriminative subspace, in-scope queries should retain class-supported residual semantic patterns. A class-conditional random Fourier support score detects residual directions that are poorly represented by the predicted intent without relying only on residual magnitude."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0294", "m_0217"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    n_classes = int(max(np.max(y), tr.logits.shape[1] - 1) + 1)

    center = x_train.mean(axis=0)
    class_means = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    for c in range(n_classes):
        if counts[c] > 0:
            class_means[c] = x_train[y == c].mean(axis=0)
        else:
            class_means[c] = center

    between = class_means - center
    _, svals, vt = np.linalg.svd(between, full_matrices=False)
    rank = int(min(64, np.sum(svals > (svals[0] * 1e-8 if svals.size else 0.0))))
    basis = vt[:rank].T

    def residual(x):
        z = x - center
        if rank:
            z = z - (z @ basis) @ basis.T
        return z

    r_train = residual(x_train)
    r_eval = residual(x_eval)
    scale = r_train.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    r_train = r_train / scale
    r_eval = r_eval / scale

    rng = np.random.default_rng(271828)
    width = 96
    w = rng.normal(size=(r_train.shape[1], width)) / np.sqrt(r_train.shape[1])
    b = rng.uniform(0.0, 2.0 * np.pi, size=width)

    def features(x):
        phase = x @ w + b
        return np.concatenate((np.cos(phase), np.sin(phase)), axis=1) / np.sqrt(width)

    phi_train = features(r_train)
    phi_eval = features(r_eval)

    prototypes = np.zeros((n_classes, phi_train.shape[1]), dtype=float)
    for c in range(n_classes):
        if counts[c] > 0:
            prototypes[c] = phi_train[y == c].mean(axis=0)

    train_sim = np.sum(phi_train * prototypes[y], axis=1)
    sim_mean = np.zeros(n_classes, dtype=float)
    sim_std = np.ones(n_classes, dtype=float)
    for c in range(n_classes):
        vals = train_sim[y == c]
        if vals.size:
            sim_mean[c] = vals.mean()
            sim_std[c] = max(vals.std(), 1e-4)

    pred = np.argmax(ev.logits, axis=1)
    pred = np.clip(pred, 0, n_classes - 1)
    eval_sim = np.sum(phi_eval * prototypes[pred], axis=1)
    scores = (sim_mean[pred] - eval_sim) / sim_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
