import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = "In-scope queries should have a concentrated angular match to one or a few normalized class prototypes. OOS queries tend to have both lower prototype evidence and more diffuse prototype assignments, so this combines prototype log-evidence with angular assignment entropy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels)

    train_unit = x_train / np.maximum(
        np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12
    )
    eval_unit = x_eval / np.maximum(
        np.linalg.norm(x_eval, axis=1, keepdims=True), 1e-12
    )

    classes, inverse = np.unique(labels, return_inverse=True)
    prototypes = np.zeros((classes.size, train_unit.shape[1]), dtype=float)
    np.add.at(prototypes, inverse, train_unit)
    counts = np.bincount(inverse, minlength=classes.size).astype(float)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    scale = 12.0
    similarities = scale * (eval_unit @ prototypes.T)
    row_max = np.max(similarities, axis=1, keepdims=True)
    exp_shifted = np.exp(similarities - row_max)
    probs = exp_shifted / np.maximum(exp_shifted.sum(axis=1, keepdims=True), 1e-12)

    log_mean_evidence = (
        row_max[:, 0]
        + np.log(np.maximum(exp_shifted.mean(axis=1), 1e-12))
    )
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    entropy /= np.log(max(classes.size, 2))

    return -log_mean_evidence + 0.2 * entropy
