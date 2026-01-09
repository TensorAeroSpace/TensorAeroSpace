## Dynamic programming / Adaptive Critic (ACD) examples

This folder contains paper-inspired **Adaptive Critic Designs** examples for `tensoraerospace.agent.ADP` on `ImprovedB747Env`.

### Notebooks (one per design)

- **`example_acd_adhdp_b747.ipynb`**: `design="adhdp"` — action-dependent scalar critic (practical TD on \(Q(s,a)\)).
- **`example_acd_hdp_b747.ipynb`**: `design="hdp"` — critic learns \(J(R)\), actor improves via model lookahead.
- **`example_acd_dhp_b747.ipynb`**: `design="dhp"` — critic learns \(\lambda=\partial J/\partial R\).
- **`example_acd_gdhp_b747.ipynb`**: `design="gdhp"` — critic learns both \(J\) and \(\lambda\) (Fig. 5).
- **`example_acd_addhp_b747.ipynb`**: `design="addhp"` — action-dependent DHP (gradients w.r.t \(R,A\)).
- **`example_acd_adgdhp_b747.ipynb`**: `design="adgdhp"` — action-dependent GDHP (Fig. 7).

### Shared helper

- **`acd_b747_common.py`**: environment creation + rollout/plots/metrics used by all notebooks.


