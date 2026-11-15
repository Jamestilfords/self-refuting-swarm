import numpy as np

FAMILY = "other"
ARM = "ce"
RATIONALE = "In-scope features should not only align with a known class prototype but should usually have a clear angular preference over competing prototypes. OOS queries can have both weak nearest-prototype alignment and a small nearest-versus-runner-up cosine margin."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = np.asarray(tr.feats, dtype=float)
    x_eval = np.asarray(ev.feats, dtype=float)
    labels = np.asarray(tr.labels, dtype=int)

    n_classes = tr.logits.shape[1]
    train_unit = x_train / np.maximum(
        np.linalg.norm(x_train, axis=1, keepdims=True), 1e-12
    )
    eval_unit = x_eval / np.maximum(
        np.linalg.norm(x_eval, axis=1, keepdims=True), 1e-12
    )

    prototypes = np.zeros((n_classes, train_unit.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, train_unit)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes /= np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12
    )

    def prototype_signals(x):
        similarities = x @ prototypes.T
        if n_classes > 1:
            top_two = np.partition(similarities, -2, axis=1)[:, -2:]
            nearest = np.max(top_two, axis=1)
            runner_up = np.min(top_two, axis=1)
            margin = nearest - runner_up
        else:
            nearest = similarities[:, 0]
            margin = np.zeros_like(nearest)
        return 1.0 - nearest, 1.0 - margin

    train_novelty, train_ambiguity = prototype_signals(train_unit)
    eval_novelty, eval_ambiguity = prototype_signals(eval_unit)

    novelty_ref = np.sort(train_novelty)
    ambiguity_ref = np.sort(train_ambiguity)

    novelty_pct = np.searchsorted(novelty_ref, eval_novelty, side="right") / float(
        len(novelty_ref)
    )
    ambiguity_pct = np.searchsorted(
        ambiguity_ref, eval_ambiguity, side="right"
    ) / float(len(ambiguity_ref))

    return 0.65 * novelty_pct + 0.35 * ambiguity_pct
