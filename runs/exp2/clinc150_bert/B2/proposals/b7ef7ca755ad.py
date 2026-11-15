import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "A query assigned to an intent should point in a similar direction to that intent's L2-normalized feature prototype. Class-calibrated angular disagreement detects samples that the classifier assigns confidently but whose representation is atypical for the predicted intent."
SOURCE = "https://arxiv.org/html/2505.18032"
PARENTS = []
MEMORY_USED = ["m_0084"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    x_tr = np.asarray(tr.feats, dtype=np.float64)
    x_ev = np.asarray(ev.feats, dtype=np.float64)
    y_tr = np.asarray(tr.labels, dtype=np.int64)

    tr_unit = x_tr / np.maximum(np.linalg.norm(x_tr, axis=1, keepdims=True), 1e-12)
    ev_unit = x_ev / np.maximum(np.linalg.norm(x_ev, axis=1, keepdims=True), 1e-12)

    n_classes = tr.logits.shape[1]
    global_proto = np.mean(tr_unit, axis=0)
    global_proto /= max(np.linalg.norm(global_proto), 1e-12)

    prototypes = np.tile(global_proto, (n_classes, 1))
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    sums = np.zeros((n_classes, tr_unit.shape[1]), dtype=np.float64)
    np.add.at(sums, y_tr, tr_unit)

    present = counts > 0
    prototypes[present] = sums[present] / counts[present, None]
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)

    train_cos = np.sum(tr_unit * prototypes[y_tr], axis=1)
    train_shortfall = 1.0 - np.clip(train_cos, -1.0, 1.0)

    class_mean = np.zeros(n_classes, dtype=np.float64)
    np.add.at(class_mean, y_tr, train_shortfall)
    class_mean[present] /= counts[present]
    class_mean[~present] = np.mean(train_shortfall)

    centered = train_shortfall - class_mean[y_tr]
    class_var = np.zeros(n_classes, dtype=np.float64)
    np.add.at(class_var, y_tr, centered * centered)
    class_std = np.sqrt(class_var / np.maximum(counts, 1.0))
    fallback_std = max(float(np.std(train_shortfall)), 1e-6)
    class_std = np.maximum(class_std, fallback_std * 0.1)

    pred = np.argmax(np.asarray(ev.logits), axis=1)
    eval_cos = np.sum(ev_unit * prototypes[pred], axis=1)
    eval_shortfall = 1.0 - np.clip(eval_cos, -1.0, 1.0)

    scores = (eval_shortfall - class_mean[pred]) / class_std[pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
