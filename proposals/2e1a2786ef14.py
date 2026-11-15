import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents tend to exhibit recurring top-two rivalry patterns: both the identity of the "
    "runner-up intent and the winning margin conditioned on that rivalry are structured in training data. "
    "Unexpected competitor pairs or atypical pair-specific margins indicate out-of-scope logit competition."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-8

    train_logits = np.asarray(tr.logits, dtype=float)
    eval_logits = np.asarray(ev.logits, dtype=float)
    n_classes = train_logits.shape[1]

    train_winner = np.argmax(train_logits, axis=1)
    eval_winner = np.argmax(eval_logits, axis=1)

    train_masked = train_logits.copy()
    train_masked[np.arange(len(train_logits)), train_winner] = -np.inf
    train_runner = np.argmax(train_masked, axis=1)
    train_margin = (
        train_logits[np.arange(len(train_logits)), train_winner]
        - train_logits[np.arange(len(train_logits)), train_runner]
    )

    eval_masked = eval_logits.copy()
    eval_masked[np.arange(len(eval_logits)), eval_winner] = -np.inf
    eval_runner = np.argmax(eval_masked, axis=1)
    eval_margin = (
        eval_logits[np.arange(len(eval_logits)), eval_winner]
        - eval_logits[np.arange(len(eval_logits)), eval_runner]
    )

    pair_count = np.zeros((n_classes, n_classes), dtype=float)
    pair_sum = np.zeros((n_classes, n_classes), dtype=float)
    pair_sumsq = np.zeros((n_classes, n_classes), dtype=float)
    winner_count = np.bincount(train_winner, minlength=n_classes).astype(float)
    winner_sum = np.bincount(
        train_winner, weights=train_margin, minlength=n_classes
    ).astype(float)
    winner_sumsq = np.bincount(
        train_winner, weights=train_margin * train_margin, minlength=n_classes
    ).astype(float)

    np.add.at(pair_count, (train_winner, train_runner), 1.0)
    np.add.at(pair_sum, (train_winner, train_runner), train_margin)
    np.add.at(pair_sumsq, (train_winner, train_runner), train_margin * train_margin)

    global_mean = float(np.mean(train_margin))
    global_var = float(np.var(train_margin))
    global_var = max(global_var, eps)

    winner_mean = (winner_sum + 8.0 * global_mean) / (winner_count + 8.0)
    winner_second = (
        winner_sumsq + 8.0 * (global_var + global_mean * global_mean)
    ) / (winner_count + 8.0)
    winner_var = np.maximum(winner_second - winner_mean * winner_mean, eps)

    prior_strength = 6.0
    pair_mean = (
        pair_sum + prior_strength * winner_mean[:, None]
    ) / (pair_count + prior_strength)
    pair_second = (
        pair_sumsq
        + prior_strength * (
            winner_var[:, None] + winner_mean[:, None] * winner_mean[:, None]
        )
    ) / (pair_count + prior_strength)
    pair_var = np.maximum(pair_second - pair_mean * pair_mean, eps)

    counts = pair_count[eval_winner, eval_runner]
    totals = winner_count[eval_winner]
    pair_surprisal = -np.log((counts + 1.0) / (totals + n_classes))

    expected_margin = pair_mean[eval_winner, eval_runner]
    margin_var = pair_var[eval_winner, eval_runner]
    margin_deviation = (eval_margin - expected_margin) ** 2 / margin_var

    return pair_surprisal + 0.35 * np.log1p(margin_deviation)
