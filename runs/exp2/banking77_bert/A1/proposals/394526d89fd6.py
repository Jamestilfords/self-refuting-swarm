import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = (
    "In-scope examples tend to exhibit class-specific Dirichlet concentration profiles, including "
    "evidence strength, predictive concentration entropy, and top-class dominance. OOS inputs can "
    "have atypical profiles even when their total evidence alone (vacuity) appears in-distribution."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def profile(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        total = np.sum(alpha, axis=1)
        probs = alpha / total[:, None]
        k = alpha.shape[1]

        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
        entropy = entropy / np.log(max(k, 2))

        top2 = np.partition(probs, -2, axis=1)[:, -2:]
        dominance = top2[:, 1] - top2[:, 0]

        return np.column_stack((
            np.log1p(total),
            entropy,
            dominance,
        ))

    z_train = profile(tr.logits)
    z_eval = profile(ev.logits)
    labels = np.asarray(tr.labels)
    classes = np.unique(labels)

    global_var = np.var(z_train, axis=0) + 1e-6
    best = np.full(z_eval.shape[0], np.inf, dtype=float)

    for c in classes:
        members = z_train[labels == c]
        mean = np.mean(members, axis=0)

        if members.shape[0] > 2:
            local_var = np.var(members, axis=0)
            weight = members.shape[0] / (members.shape[0] + 10.0)
            var = weight * local_var + (1.0 - weight) * global_var
        else:
            var = global_var

        var = np.maximum(var, 1e-6)
        diff = z_eval - mean
        dist = np.sum((diff * diff) / var, axis=1)
        best = np.minimum(best, dist)

    return np.maximum(best, 0.0)
