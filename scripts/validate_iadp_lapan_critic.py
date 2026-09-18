"""Test iADP's Bellman fixed point on exact nominal LAPAN LQR samples.

This is an algebraic oracle with privileged model knowledge, not a flight trial.
The reference is constant per sample. States are sampled independently so all
15 symmetric features on the 5-dimensional active subspace can be identified.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=391)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    import sys
    import numpy as np
    from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

    sys.path.insert(0, str(args.repo.resolve()))
    from tensoraerospace.aerospacemodel.lapan import LAPAN
    from tensoraerospace.agent.iadp import IADPAgent, IADPConfig

    model = LAPAN(np.zeros(4), 5, dt=0.02)
    F = np.zeros((5, 5))
    F[:4, :4] = model.filt_A
    F[-1, -1] = 1
    G = np.vstack([model.filt_B * np.pi / 180, [[0]]])
    R = np.array([[(abs(G[2, 0] / 0.02) / 20) ** 2]])
    c = np.array([0, 0, 1, 0, -1])
    Q = np.outer(c, c)
    gamma = 0.99
    P = solve_discrete_are(np.sqrt(gamma) * F, np.sqrt(gamma) * G, Q, R)
    K = np.linalg.solve(R + gamma * G.T @ P @ G, gamma * G.T @ P @ F)
    Acl = F - G @ K
    lyap = solve_discrete_lyapunov(np.sqrt(gamma) * Acl.T, Q + K.T @ R @ K)
    idx = [0, 1, 2, 3, 6]
    fullP = np.zeros((8, 8))
    fullP[np.ix_(idx, idx)] = P
    states = np.random.default_rng(args.seed).normal(size=(300, 5)) * [
        0.1,
        0.05,
        0.005,
        0.01,
        0.01,
    ]
    records = []
    for ridge in [0, 1e-10, 1e-4]:
        for enforce_psd in [False, True]:
            a = IADPAgent(
                4,
                1,
                IADPConfig(
                    Q=np.diag([0, 0, 1, 0]),
                    R=R,
                    gamma=gamma,
                    P_init=fullP,
                    policy_eval_regularization=ridge,
                    policy_eval_window=300,
                    enforce_psd=enforce_psd,
                ),
            )
            for x in states:
                u = -K @ x
                xn = F @ x + G @ u
                X = np.zeros(8)
                X[idx] = x
                Xn = np.zeros(8)
                Xn[idx] = xn
                a._window.append(dict(X=X, Xnext=Xn, cost=float(x @ Q @ x + u @ R @ u)))
            features = np.array([np.outer(s["X"], s["X"]).ravel() for s in a._window])
            a._policy_evaluation()
            learned = a.P[np.ix_(idx, idx)]
            kl = np.linalg.solve(
                R + gamma * G.T @ learned @ G, gamma * G.T @ learned @ F
            )
            records.append(
                dict(
                    ridge=ridge,
                    enforce_psd=enforce_psd,
                    relative_P_error=float(
                        np.linalg.norm(learned - P) / np.linalg.norm(P)
                    ),
                    relative_K_error=float(np.linalg.norm(kl - K) / np.linalg.norm(K)),
                    physical_closed_loop_radius=float(
                        max(abs(np.linalg.eigvals(F[:4, :4] - G[:4] @ kl[:, :4])))
                    ),
                )
            )
    result = dict(
        disclaimer="Privileged exact nominal model and synthetic on-policy samples. Algebraic oracle, not flight training.",
        lyapunov_DARE_relative_error=float(
            np.linalg.norm(P - lyap) / np.linalg.norm(P)
        ),
        feature_rank=int(np.linalg.matrix_rank(features)),
        active_variables=5,
        independent_symmetric_features=15,
        records=records,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
