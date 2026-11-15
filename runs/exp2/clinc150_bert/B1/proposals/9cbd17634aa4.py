import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "In-scope normalized bottleneck features should have both strong alignment to one intent "
    "prototype and a clear cosine gap from competing prototypes; OOS queries often violate either condition."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x_train = ctx.h.l2(tr.feats)
    x_eval = ctx.h.l2(ev.feats)
    labels = tr.labels.astype(int)
    n_classes = int(labels.max()) + 1

    prototypes = np.zeros((n_classes, x_train.shape[1]), dtype=float)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    np.add.at(prototypes, labels, x_train)
    prototypes /= np.maximum(counts[:, None], 1.0)
    prototypes = ctx.h.l2(prototypes)

    train_cos = x_train @ prototypes.T
    eval_cos = x_eval @ prototypes.T

    train_top2 = np.partition(train_cos, -2, axis=1)[:, -2:]
    eval_top2 = np.partition(eval_cos, -2, axis=1)[:, -2:]

    train_best = train_top2[:, 1]
    train_margin = train_top2[:, 1] - train_top2[:, 0]
    eval_best = eval_top2[:, 1]
    eval_margin = eval_top2[:, 1] - eval_top2[:, 0]

    best_scale = np.std(train_best) + 1e-6
    margin_scale = np.std(train_margin) + 1e-6
    best_deficit = (np.mean(train_best) - eval_best) / best_scale
    margin_deficit = (np.mean(train_margin) - eval_margin) / margin_scale

    return best_deficit + 0.35 * margin_deficit
