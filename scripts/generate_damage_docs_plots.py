"""Generate illustration plots for the damage modeling docs.

Produces five PNG figures used in:
- docs/{en,ru}/model/img/  — geometry, strip-theory, baseline-vs-damaged
- docs/{en,ru}/cookbook/img/ — ET-DHP damage cookbook hero plot

Run from the repo root::

    poetry run python scripts/generate_damage_docs_plots.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
MODEL_IMG_EN = REPO / "docs" / "en" / "model" / "img"
MODEL_IMG_RU = REPO / "docs" / "ru" / "model" / "img"
COOKBOOK_IMG_EN = REPO / "docs" / "en" / "cookbook" / "img"
COOKBOOK_IMG_RU = REPO / "docs" / "ru" / "cookbook" / "img"

for d in (MODEL_IMG_EN, MODEL_IMG_RU, COOKBOOK_IMG_EN, COOKBOOK_IMG_RU):
    d.mkdir(parents=True, exist_ok=True)


def save_dual(fig, name: str, *, target: str = "model"):
    """Save fig to both en/ and ru/ image dirs."""
    if target == "model":
        out_dirs = [MODEL_IMG_EN, MODEL_IMG_RU]
    else:
        out_dirs = [COOKBOOK_IMG_EN, COOKBOOK_IMG_RU]
    for d in out_dirs:
        path = d / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"  → {path.relative_to(REPO)}")
    plt.close(fig)


# ============================================================
# Figure 1: F-16 section breakdown (top-down view)
# ============================================================
def plot_section_layout():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    geo = load_f16_geometry()

    fig, ax = plt.subplots(figsize=(11, 6))
    type_colors = {
        "wing": "#4a90e2",
        "stab": "#e2884a",
        "vtail": "#9b59b6",
        "control": "#27ae60",
        "fuselage": "#95a5a6",
    }
    legend_seen: set[str] = set()
    for s in geo.sections:
        # Top-down: x = body-x (forward+), y = body-y (right+)
        # Approximate each section as a rectangle centred at (cg_x, span_pos)
        # with extent ≈ chord × (something proportional to area/chord).
        if s.area > 0:
            cx = s.aero_x_arm if abs(s.aero_x_arm) > 1e-3 else s.cg_local[0]
            cy = s.span_position
            chord = max(s.chord, 0.5)
            span = max(s.area / chord, 0.4)
            color = type_colors[s.type]
            label = s.type if s.type not in legend_seen else None
            if label is not None:
                legend_seen.add(s.type)
            rect = mpatches.Rectangle(
                (cx - chord / 2, cy - span / 2), chord, span,
                facecolor=color, edgecolor="black", alpha=0.7, linewidth=0.8,
                label=label,
            )
            ax.add_patch(rect)
            ax.text(cx, cy, s.name, ha="center", va="center",
                    fontsize=7, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.6))
        else:
            # Fuselage — draw as a long thin oval from nose to tail
            ax.add_patch(mpatches.Ellipse(
                (0.0, 0.0), 8.0, 0.8,
                facecolor=type_colors[s.type], edgecolor="black",
                alpha=0.4, linewidth=0.8,
                label=s.type if s.type not in legend_seen else None,
            ))
            legend_seen.add(s.type)
            ax.text(0.0, 0.0, "fuselage_main", ha="center", va="center",
                    fontsize=8, color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.6))

    # Aircraft CG marker at origin
    ax.plot(0.0, 0.0, "rx", markersize=14, markeredgewidth=2,
            label="aircraft CG (origin)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)
    ax.axvline(0.0, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)

    ax.set_xlabel("body-x (m)  →  forward")
    ax.set_ylabel("body-y (m)  →  right")
    ax.set_title("F-16 section layout (top-down view)")
    ax.set_aspect("equal")
    ax.set_xlim(-7.0, 4.0)
    ax.set_ylim(-5.5, 5.5)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    return fig


# ============================================================
# Figure 2: Aero corrections — delta_cy vs alpha for different damage levels
# ============================================================
def plot_strip_theory():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import aero_corrections
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )

    geo = load_f16_geometry()
    alpha_arr = np.linspace(-5, 15, 100)  # degrees
    alpha_rad = np.radians(alpha_arr)

    fractions = [0.0, 0.30, 0.60, 1.0]
    colors = ["#27ae60", "#3498db", "#e67e22", "#c0392b"]
    labels = [
        "healthy (f=0)",
        "30% bilateral tip loss",
        "60% bilateral tip loss",
        "100% bilateral tip loss",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel A: Symmetric — delta_cy vs alpha
    for f, c, lab in zip(fractions, colors, labels):
        state = DamageState.healthy(geo)
        state.set_section_loss("left_tip", f)
        state.set_section_loss("right_tip", f)
        dcy = np.array([
            aero_corrections.delta_cy(a, 0.0, geo, state) for a in alpha_rad
        ])
        axes[0].plot(alpha_arr, dcy, color=c, linewidth=2, label=lab)
    axes[0].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[0].set_xlabel(r"angle of attack $\alpha$, deg")
    axes[0].set_ylabel(r"$\Delta C_y$ (lift coefficient delta)")
    axes[0].set_title("Symmetric tip loss → reduced lift")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel B: Asymmetric — delta_mx (roll moment) vs alpha
    asym_fractions = [0.0, 0.30, 0.60, 1.0]
    for f, c, lab in zip(asym_fractions, colors, labels):
        state = DamageState.healthy(geo)
        state.set_section_loss("left_tip", f)  # left only — asymmetric
        dmx = np.array([
            aero_corrections.delta_mx(a, 0.0, geo, state) for a in alpha_rad
        ])
        axes[1].plot(alpha_arr, dmx, color=c, linewidth=2,
                     label=lab.replace("bilateral", "left-only"))
    axes[1].axhline(0.0, color="black", linewidth=0.5, alpha=0.5)
    axes[1].set_xlabel(r"angle of attack $\alpha$, deg")
    axes[1].set_ylabel(r"$\Delta C_{Mx}$ (roll moment coefficient delta)")
    axes[1].set_title("Asymmetric (left-only) tip loss → roll moment")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(
        "Strip-theory aero corrections from $\\mathit{DamageState}$",
        fontsize=13,
    )
    plt.tight_layout()
    return fig


# ============================================================
# Figure 3: Mass / area / inertia recompute under symmetric loss
# ============================================================
def plot_recompute_curves():
    from tensoraerospace.aerospacemodel.f16.nonlinear.angular.params import (
        F16AngularParameters,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.presets import (
        load_f16_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.recompute import (
        recompute_inertia, recompute_mass_geometry,
    )
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage.state import (
        DamageState,
    )

    geo = load_f16_geometry()
    base_p = F16AngularParameters()
    fractions = np.linspace(0.0, 1.0, 11)

    sym_m, sym_S, sym_b, sym_jx, sym_jxy = [], [], [], [], []
    asy_m, asy_S, asy_b, asy_jx, asy_jxy = [], [], [], [], []
    asy_cgy = []
    for f in fractions:
        # Symmetric: lose both tips equally
        s_sym = DamageState.healthy(geo)
        s_sym.set_section_loss("left_tip", f)
        s_sym.set_section_loss("right_tip", f)
        mg = recompute_mass_geometry(geo, s_sym)
        inertia = recompute_inertia(geo, s_sym, cg=mg["cg"])
        sym_m.append(mg["m"])
        sym_S.append(mg["S"])
        sym_b.append(mg["b"])
        sym_jx.append(inertia["Jx"])
        sym_jxy.append(inertia["Jxy"])

        # Asymmetric: lose only left
        s_asy = DamageState.healthy(geo)
        s_asy.set_section_loss("left_tip", f)
        mg = recompute_mass_geometry(geo, s_asy)
        inertia = recompute_inertia(geo, s_asy, cg=mg["cg"])
        asy_m.append(mg["m"])
        asy_S.append(mg["S"])
        asy_b.append(mg["b"])
        asy_jx.append(inertia["Jx"])
        asy_jxy.append(inertia["Jxy"])
        asy_cgy.append(mg["cg"][1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(fractions, np.array(sym_m) / base_p.m, "-o",
                    color="#3498db", label="symmetric (both tips)")
    axes[0, 0].plot(fractions, np.array(asy_m) / base_p.m, "-s",
                    color="#e74c3c", label="asymmetric (left only)")
    axes[0, 0].set_xlabel("loss fraction f")
    axes[0, 0].set_ylabel("m / m_baseline")
    axes[0, 0].set_title("Total mass")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(fractions, np.array(sym_S) / base_p.S, "-o",
                    color="#3498db", label="symmetric")
    axes[0, 1].plot(fractions, np.array(asy_S) / base_p.S, "-s",
                    color="#e74c3c", label="asymmetric")
    axes[0, 1].set_xlabel("loss fraction f")
    axes[0, 1].set_ylabel("S / S_baseline")
    axes[0, 1].set_title("Effective wing area")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(fractions, np.array(sym_jx) / base_p.Jx, "-o",
                    color="#3498db", label="symmetric")
    axes[1, 0].plot(fractions, np.array(asy_jx) / base_p.Jx, "-s",
                    color="#e74c3c", label="asymmetric")
    axes[1, 0].set_xlabel("loss fraction f")
    axes[1, 0].set_ylabel("Jx / Jx_baseline")
    axes[1, 0].set_title("Roll-axis inertia (Huygens-Steiner)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(fractions, asy_cgy, "-s", color="#e74c3c",
                    label="asymmetric (left only)")
    axes[1, 1].axhline(0.0, color="#3498db", linestyle="--",
                       label="symmetric (both → cg=0)")
    axes[1, 1].set_xlabel("loss fraction f")
    axes[1, 1].set_ylabel("cg_y, m  (positive → right)")
    axes[1, 1].set_title("CG lateral shift")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle(
        "Parameter recompute as a function of wing-tip loss fraction",
        fontsize=13,
    )
    plt.tight_layout()
    return fig


# ============================================================
# Figure 4: Damage event timeline
# ============================================================
def plot_event_timeline():
    fig, ax = plt.subplots(figsize=(11, 3.2))

    events = [
        (5.0, "ELEVATOR_JAM_NEUTRAL", "#e67e22"),
        (10.0, "WING_STRIKE_LEFT_TIP", "#c0392b"),
        (15.0, "BIRDSTRIKE_COMPOUND", "#8e44ad"),
        (25.0, "ENGINE_FLAMEOUT", "#2c3e50"),
    ]

    ax.axhline(0.5, color="gray", linewidth=2, alpha=0.5)
    for t, name, c in events:
        ax.axvline(t, ymin=0.05, ymax=0.95, color=c, linewidth=2)
        ax.scatter([t], [0.5], color=c, s=120, zorder=3, edgecolor="black")
        ax.annotate(
            f"t = {t:.0f}s\n{name}", xy=(t, 0.5),
            xytext=(t, 0.85 if events.index((t, name, c)) % 2 == 0 else 0.15),
            ha="center", fontsize=9,
            arrowprops=dict(arrowstyle="-", color=c, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.3", facecolor=c, alpha=0.2),
        )

    ax.set_xlim(-2, 32)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time, s")
    ax.set_yticks([])
    ax.set_title("Sample DamageProfile timeline (multiple events scheduled)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================
# Figure 5: Demo trajectory — wing-tip loss in real-time simulation
# ============================================================
def plot_demo_trajectory():
    from tensoraerospace.aerospacemodel.f16.nonlinear.damage import (
        WING_STRIKE_LEFT_TIP,
    )
    from tensoraerospace.envs.f16.nonlinear_angular import NonlinearAngularF16

    n_steps = 1500
    dt = 0.01

    # Healthy run
    env_h = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=n_steps,
        dt=dt, airspeed=200.0, split_stab=True,
    )
    env_h.reset()
    log_h = {"alpha": [], "wx": [], "wz": [], "stab": []}
    for _ in range(n_steps - 1):
        obs, *_ = env_h.step(np.zeros(4))
        log_h["alpha"].append(np.degrees(obs[0]))
        log_h["wx"].append(np.degrees(obs[2]))
        log_h["wz"].append(np.degrees(obs[4]))
        log_h["stab"].append(np.degrees(obs[8]))

    # Damaged run — same scenario but with WING_STRIKE_LEFT_TIP
    env_d = NonlinearAngularF16(
        initial_state=np.zeros(14), number_time_steps=n_steps,
        dt=dt, airspeed=200.0, split_stab=True,
        damage_profile=WING_STRIKE_LEFT_TIP,
    )
    env_d.reset()
    log_d = {"alpha": [], "wx": [], "wz": [], "stab": []}
    damage_t = None
    for k in range(n_steps - 1):
        obs, _, _, _, info = env_d.step(np.zeros(4))
        if info.get("damage_events_triggered") and damage_t is None:
            damage_t = k * dt
        log_d["alpha"].append(np.degrees(obs[0]))
        log_d["wx"].append(np.degrees(obs[2]))
        log_d["wz"].append(np.degrees(obs[4]))
        log_d["stab"].append(np.degrees(obs[8]))

    t = np.arange(len(log_h["alpha"])) * dt

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    panels = [
        ("alpha", "α (angle of attack), deg"),
        ("wx", "$\\omega_x$ (roll rate), deg/s"),
        ("wz", "$\\omega_z$ (pitch rate), deg/s"),
        ("stab", "stab (deflection), deg"),
    ]
    for (key, title), ax in zip(panels, axes.flat):
        ax.plot(t, log_h[key], color="#27ae60", linewidth=1.5,
                label="healthy")
        ax.plot(t, log_d[key], color="#c0392b", linewidth=1.5,
                label="WING_STRIKE_LEFT_TIP")
        if damage_t is not None:
            ax.axvline(damage_t, color="red", linestyle="--", alpha=0.4,
                       label=f"damage @ t={damage_t:.1f}s")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    axes[1, 0].set_xlabel("time, s")
    axes[1, 1].set_xlabel("time, s")
    fig.suptitle(
        "Healthy vs damaged trajectory under zero command "
        "(WING_STRIKE_LEFT_TIP at t=10 s)", fontsize=13,
    )
    plt.tight_layout()
    return fig


# ============================================================
# Run all
# ============================================================
def main():
    print("Generating damage docs plots...")

    print("\n[1/5] F-16 section layout")
    fig = plot_section_layout()
    save_dual(fig, "damage_section_layout.png", target="model")

    print("\n[2/5] Strip-theory aero corrections")
    fig = plot_strip_theory()
    save_dual(fig, "damage_strip_theory.png", target="model")

    print("\n[3/5] Parameter recompute curves")
    fig = plot_recompute_curves()
    save_dual(fig, "damage_recompute_curves.png", target="model")

    print("\n[4/5] DamageProfile timeline example")
    fig = plot_event_timeline()
    save_dual(fig, "damage_event_timeline.png", target="model")

    print("\n[5/5] Demo trajectory: healthy vs damaged")
    fig = plot_demo_trajectory()
    save_dual(fig, "damage_demo_trajectory.png", target="model")

    print("\nAll plots saved.")


if __name__ == "__main__":
    main()
