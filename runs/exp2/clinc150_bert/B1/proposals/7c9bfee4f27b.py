import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "Within each predicted intent, in-scope examples tend to have a characteristic separation between their top two logits. Queries with unusually small predicted-class-conditioned margins are likely ambiguous or outside the learned intent regions."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    train_logits = np.nan_to_num(
        np.asarray(tr.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_logits = np.nan_to_num(
        np.asarray(ev.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )

    def margins(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        return top2[:, 1] - top2[:, 0]

    train_margin = margins(train_logits)
    eval_margin = margins(eval_logits)
    train_pred = np.argmax(train_logits, axis=1)
    eval_pred = np.argmax(eval_logits, axis=1)

    k = train_logits.shape[1]
    global_mean = float(np.mean(train_margin))
    global_std = float(np.std(train_margin))
    global_std = max(global_std, 1e-6)

    means = np.full(k, global_mean, dtype=float)
    stds = np.full(k, global_std, dtype=float)
    for c in range(k):
        values = train_margin[train_pred == c]
        if values.size >= 5:
            means[c] = float(np.mean(values))
            stds[c] = max(float(np.std(values)), 1e-6)

    return (means[eval_pred] - eval_margin) / stds[eval_pred]
