import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The gap between Shannon and collision entropy of normalized Dirichlet evidence measures "
    "how evidence is distributed among competing intents, beyond total evidence or vacuity. "
    "A predicted-intent-conditioned residual from the training strength-shape relationship flags unusual evidential competition."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0050", "m_0092"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-8)
        evidence = np.maximum(alpha - 1.0, 0.0)
        total_evidence = evidence.sum(axis=1)

        # If evidence is nearly absent, alpha itself remains a well-defined allocation.
        use_alpha = total_evidence <= 1e-10
        mass = evidence.copy()
        if np.any(use_alpha):
            mass[use_alpha] = alpha[use_alpha]

        mass = mass / np.maximum(mass.sum(axis=1, keepdims=True), 1e-12)
        mass = np.clip(mass, 1e-12, 1.0)

        shannon = -np.sum(mass * np.log(mass), axis=1)
        collision = -np.log(np.sum(mass * mass, axis=1))
        renyi_gap = shannon - collision
        leader_share = np.max(mass, axis=1)
        strength = np.log1p(total_evidence)
        pred = np.argmax(logits, axis=1)
        return np.column_stack((renyi_gap, leader_share)), strength, pred

    x_tr, s_tr, p_tr = shape_features(tr.logits)
    x_ev, s_ev, p_ev = shape_features(ev.logits)
    n_classes = tr.logits.shape[1]

    # Global ridge regressions provide stable fallbacks for rarely predicted classes.
    design_global = np.column_stack((np.ones(len(s_tr)), s_tr))
    ridge = np.diag([1e-6, 1e-4])
    beta_global = np.linalg.solve(
        design_global.T @ design_global + ridge,
        design_global.T @ x_tr,
    )
    resid_global = x_tr - design_global @ beta_global
    cov_global = np.cov(resid_global, rowvar=False) if len(resid_global) > 2 else np.eye(2)
    cov_global = np.atleast_2d(cov_global) + 1e-5 * np.eye(2)
    inv_global = np.linalg.pinv(cov_global)

    scores = np.empty(len(s_ev), dtype=float)
    for c in range(n_classes):
        train_mask = p_tr == c
        eval_mask = p_ev == c
        if not np.any(eval_mask):
            continue

        if np.sum(train_mask) >= 12:
            dc = np.column_stack((np.ones(np.sum(train_mask)), s_tr[train_mask]))
            bc = np.linalg.solve(
                dc.T @ dc + ridge,
                dc.T @ x_tr[train_mask],
            )
            rc = x_tr[train_mask] - dc @ bc
            cov = np.cov(rc, rowvar=False) if len(rc) > 2 else cov_global
            cov = np.atleast_2d(cov) + 1e-5 * np.eye(2)
            inv_cov = np.linalg.pinv(cov)
            beta = bc
        else:
            beta = beta_global
            inv_cov = inv_global

        de = np.column_stack((np.ones(np.sum(eval_mask)), s_ev[eval_mask]))
        residual = x_ev[eval_mask] - de @ beta
        scores[eval_mask] = np.einsum("ij,jk,ik->i", residual, inv_cov, residual)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
