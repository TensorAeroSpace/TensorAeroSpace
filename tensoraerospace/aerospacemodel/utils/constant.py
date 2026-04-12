"""Constants for displaying states and control signals.

This module contains dictionaries with LaTeX formatting for displaying names
of states and control signals in plots and documentation. Both Russian and English
notations are supported for various aircraft parameters.

Available dictionaries:
- state_to_latex_rus: Russian state notations
- state_to_latex_eng: English state notations
- ref_state_to_latex_rus: Russian reference state notations
- ref_state_to_latex_eng: English reference state notations
- control_to_latex_rus: Russian control signal notations
- control_to_latex_eng: English control signal notations
"""

state_to_latex_rus = {
    "alpha": r"$\alpha$" + ", deg.",
    "beta": r"$\beta$" + ", deg.",
    "wx": r"$\omega_x$" + ", deg./s",
    "wy": r"$\omega_y$" + ", deg./s",
    "wz": r"$\omega_z$" + ", deg./s",
    # English state names
    "q": r"$\omega_z$" + ", deg./s",
    "u": r"$u$" + ", m/s",
    "w": r"$w$" + ", m/s",
    "v": r"$v$" + ", m/s",
    "p": r"$\omega_x$" + ", deg./s",
    "r": r"$\omega_y$" + ", deg./s",
    "gamma": r"$\gamma$" + ", deg.",
    "phi": r"$\gamma$" + ", deg.",
    "psi": r"$\varpsi$" + ", deg.",
    "theta": r"$\vartheta$" + ", deg.",
    "stab": r"$\delta_{B}$" + ", deg.",
    "ele": r"$\delta_{B}$" + ", deg.",
    "ail": r"$\delta_{A}$" + ", deg.",
    "dir": r"$\delta_{R}$" + ", deg.",
    "rud": r"$\delta_{R}$" + ", deg.",
    "dstab": r"$\dot{\delta_{B}}$" + ", deg./s",
    "dail": r"$\dot{\delta_{A}}$" + ", deg./s",
    "ddir": r"$\dot{\delta_{R}}$" + ", deg./s",
    "altitude": r"Altitude" + ", m.",
    "h": r"$h$" + ", m.",
    "rho": r"$\rho$",
    "rho_dot": r"$\dot{\rho}$" + ", m/s",
    "theta_dot": r"$\dot{\theta}$" + ", rad/s",
    "omega": r"$\omega$" + ", rad/s",
}

ref_state_to_latex_rus = {
    "alpha": r"$\alpha^{ref}$" + ", deg.",
    "beta": r"$\beta^{ref}$" + ", deg.",
    "wx": r"$\omega^{ref}_x$" + ", deg./s",
    "wy": r"$\omega^{ref}_y$" + ", deg./s",
    "wz": r"$\omega^{ref}_z$" + ", deg./s",
    "q": r"$\omega^{ref}_z$" + ", deg./s",
    "u": r"$u^{ref}$" + ", m/s",
    "w": r"$w^{ref}$" + ", m/s",
    "v": r"$v^{ref}$" + ", m/s",
    "p": r"$\omega^{ref}_x$" + ", deg./s",
    "r": r"$\omega^{ref}_y$" + ", deg./s",
    "gamma": r"$\gamma^{ref}$" + ", deg.",
    "phi": r"$\gamma^{ref}$" + ", deg.",
    "psi": r"$\varpsi^{ref}$" + ", deg.",
    "theta": r"$\vartheta^{ref}$" + ", deg.",
    "stab": r"$\delta^{ref}_{B}$" + ", deg.",
    "ele": r"$\delta{ref}_{B}$" + ", deg.",
    "ail": r"$\delta{ref}_{Э}$" + ", deg.",
    "dir": r"$\delta{ref}_{Н}$" + ", deg.",
    "rud": r"$\delta{ref}_{Н}$" + ", deg.",
    "altitude": r"Высота" + ", м.",
    "h": r"$h^{ref}$" + ", м.",
    "rho": r"$\rho^{ref}$",
    "rho_dot": r"$\dot{\rho}^{ref}$" + ", m/s",
    "theta_dot": r"$\dot{\theta}^{ref}$" + ", rad/s",
    "omega": r"$\omega^{ref}$" + ", rad/s",
}

ref_state_to_latex_eng = {
    "alpha": r"$\alpha^{ref}$" + ", deg",
    "beta": r"$\beta^{ref}$" + ", deg.",
    "wx": r"$\omega^{ref}_x$" + ", deg./s",
    "wy": r"$\omega^{ref}_y$" + ", deg./s",
    "wz": r"$\omega^{ref}_z$" + ", deg./s",
    "q": r"$\omega^{ref}_z$" + ", deg./s",
    "u": r"$u^{ref}$" + ", m/s",
    "w": r"$w^{ref}$" + ", m/s",
    "v": r"$v^{ref}$" + ", m/s",
    "p": r"$\omega^{ref}_x$" + ", deg./s",
    "r": r"$\omega^{ref}_y$" + ", deg./s",
    "gamma": r"$\gamma^{ref}$" + ", deg.",
    "phi": r"$\gamma^{ref}$" + ", deg.",
    "psi": r"$\varpsi^{ref}$" + ", deg.",
    "theta": r"$\vartheta^{ref}$" + ", deg.",
    "stab": r"$\delta^{ref}_{S_{act}}$" + ", deg.",
    "ele": r"$\delta^{ref}_{S_{act}}$" + ", deg.",
    "ail": r"$\delta^{ref}_{A_{act}}$" + ", deg.",
    "dir": r"$\delta^{ref}_{D_{act}}$" + ", deg.",
    "rud": r"$\delta^{ref}_{D_{act}}$" + ", deg.",
    "h": r"$h^{ref}$" + ", m",
    "rho": r"$\rho^{ref}$",
    "rho_dot": r"$\dot{\rho}^{ref}$" + ", m/s",
    "theta_dot": r"$\dot{\theta}^{ref}$" + ", rad/s",
    "omega": r"$\omega^{ref}$" + ", rad/s",
}


state_to_latex_eng = {
    "alpha": r"$\alpha$" + ", deg",
    "beta": r"$\beta$" + ", deg.",
    "wx": r"$\omega_x$" + ", deg./s",
    "wy": r"$\omega_y$" + ", deg./s",
    "wz": r"$\omega_z$" + ", deg./s",
    "q": r"$\omega_z$" + ", deg./s",
    "u": r"$u$" + ", m/s",
    "w": r"$w$" + ", m/s",
    "v": r"$v$" + ", m/s",
    "p": r"$\omega_x$" + ", deg./s",
    "r": r"$\omega_y$" + ", deg./s",
    "gamma": r"$\gamma$" + ", deg.",
    "phi": r"$\gamma$" + ", deg.",
    "psi": r"$\varpsi$" + ", deg.",
    "theta": r"$\vartheta$" + ", deg.",
    "stab": r"$\delta_{S_{act}}$" + ", deg.",
    "ele": r"$\delta_{S_{act}}$" + ", deg.",
    "ail": r"$\delta_{A_{act}}$" + ", deg.",
    "dir": r"$\delta_{D_{act}}$" + ", deg.",
    "rud": r"$\delta_{D_{act}}$" + ", deg.",
    "dstab": r"$\dot{\delta_{D}}$" + ", deg./s",
    "dail": r"$\dot{\delta_{A}}$" + ", deg./s",
    "ddir": r"$\dot{\delta_{D}}$" + ", deg./s",
    "h": r"$h$" + ", m",
    "rho": r"$\rho$",
    "rho_dot": r"$\dot{\rho}$" + ", m/s",
    "theta_dot": r"$\dot{\theta}$" + ", rad/s",
    "omega": r"$\omega$" + ", rad/s",
}

control_to_latex_rus = {
    "stab": r"$\delta_{S_{act}}$" + ", deg.",
    "ele": r"$\delta_{S_{act}}$" + ", deg.",
    "ail": r"$\delta_{A_{act}}$" + ", deg.",
    "dir": r"$\delta_{D_{act}}$" + ", deg.",
    "rud": r"$\delta_{D_{act}}$" + ", deg.",
    "u2": r"$u_2$" + ", N",
    "delta_t": r"$\delta_T$",
    "thrust": r"$T$" + ", N",
}

control_to_latex_eng = {
    "stab": r"$\delta_{B_{act}}$" + ", deg.",
    "ele": r"$\delta_{B_{act}}$" + ", deg.",
    "ail": r"$\delta_{Э_{act}}$" + ", deg.",
    "dir": r"$\delta_{Н_{act}}$" + ", deg.",
    "rud": r"$\delta_{Н_{act}}$" + ", deg.",
    "u2": r"$u_2$" + ", N",
    "delta_t": r"$\delta_T$",
    "thrust": r"$T$" + ", N",
}
