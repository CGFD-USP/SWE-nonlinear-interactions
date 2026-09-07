"""Investigative (non-paper) script: does the quartet's period shift away
from Quartet C's RH-only triad period grow with the Kelvin-like mode
EG(1,1)'s own driving velocity?

Motivation: figure 8 of the paper (``examples/figures/paper_figure008_
quartet_rossby_kelvin_panel.py``, Quartet C = RH(4,5)+RH(3,4)+RH(1,2)+
EG(1,1), all four modes driven at 40m/s) shows the quartet's own dominant
period (the spectrum panel's black "Quartet" peak) sitting to the right of
Triad 1's own period (the blue "Triad 1" peak, RH-only, ~3.8d) -- a ~4.55d
period entirely absent from either constituent triad alone. Checking the
dissertation's Table 3.3 example 2 (EG(1,1) driven at 80m/s, found in this
session's own conversation) reproduced a similarly shifted (~7.7d) period.
80m/s is not a realistic gravity-wave velocity, though -- this script
instead sweeps the Kelvin mode's velocity over a physically-plausible
range (10-100m/s, step 10) at Quartet C's own figure-8 baseline (RH(4,5)/
RH(3,4)/RH(1,2) held at 40m/s each, matching the published figure, only
EG(1,1)'s own velocity varied) and tests the conjecture that the period
shift grows monotonically with it.

Shift definition, matching the paper figure's own visual comparison ("the
horizontal difference between the blue peak around 4 days and the new
quartet peak"): shift = (full quartet's own novelty period, i.e. the peak
absent from every containing sub-triad, ``rsw_sphere.dynamics.
diagnostics_report.compute_diagnostics_report``'s own ``novelty_period_
final_days``) minus (Triad 1's own dominant period, RH-only). Both are
computed per Rossby mode (RH(4,5)/RH(3,4)/RH(1,2)) -- all three share the
same triad frequency structure, so the three shift curves are expected to
overlap; plotting all three both confirms this and catches any anomaly.

Outputs (outputs/utilities/quartet_kelvin_velocity_period_shift/):
    period_shift_vs_velocity.png -- shift (days) vs EG(1,1) velocity (m/s)
    spectra_grid.png -- one spectrum panel per swept velocity (Triad 1 vs
        Quartet, RH(4,5)), so the identified shift can be checked by eye
        against the same visual the paper figure itself uses.

Run:

    python examples/utilities/quartet_kelvin_velocity_period_shift.py
"""
import dataclasses
import os

import _bootstrap  # noqa: F401 -- repo root on sys.path

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.trajectory_cache import _mode_slug
from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report
from rsw_sphere.utilities.periods import _power_spectrum
from rsw_sphere.plotting.style import apply_house_style

WAVE_SET_KEY = "quartet_rossby_kelvin"
BASELINE_ABC = (40.0, 40.0, 40.0)  # RH(4,5), RH(3,4), RH(1,2) -- figure 8's own baseline
KELVIN_VELOCITIES = list(range(10, 101, 10))  # EG(1,1), m/s
TF_DAYS = 200.0
H = 0.01
ROSSBY_LABELS = ["RH(4,5)", "RH(3,4)", "RH(1,2)"]
OUT_DIR = os.path.join(_bootstrap.ROOT, "outputs", "utilities", "quartet_kelvin_velocity_period_shift")


def _run_one(v_kelvin: float):
    """Full quartet + both constituent triads at (40,40,40,v_kelvin)."""
    from run_dynamics import run_dynamics
    spec0 = RunConfig.from_registry_entry(WAVE_SET_KEY).wave_set_spec
    spec = dataclasses.replace(spec0, velocities=BASELINE_ABC + (v_kelvin,))
    config = RunConfig.from_wave_set(spec, tf_days=TF_DAYS, h=H, plot=False, parallel=False)
    results = run_dynamics(config)
    return spec, results


def _triad1_name(spec):
    """Name of the RH-only constituent triad (a,b,c), matching run_dynamics's
    own 'triad_<slug>_<slug>' unit naming (paper_figure008's own convention)
    -- the constituent triad whose two members are *not* EG(1,1).
    """
    for i in range(spec.n_triads()):
        member_p, member_q, _ = spec.sub_triad_modes(i)
        if (1, 1, 1) in (member_p, member_q):
            continue
        return f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
    raise ValueError("could not identify the RH-only constituent triad")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    apply_house_style(base_size=14)

    shifts = {label: [] for label in ROSSBY_LABELS}
    spectra_cache = []  # (v_kelvin, spec, results) for the grid plot

    print(f"{'v_kelvin':>9}  " + "  ".join(f"{lab:>10}" for lab in ROSSBY_LABELS))
    for v in KELVIN_VELOCITIES:
        spec, results = _run_one(v)
        triad1 = _triad1_name(spec)
        report = compute_diagnostics_report(results, spec)

        row = []
        for label in ROSSBY_LABELS:
            triad1_period = report["per_mode_unit"][triad1][label]["period_global"]
            final = next(d for d in report["final"] if d["mode"] == label)
            novelty_period = final["novelty_period_final_days"]
            shift = novelty_period - triad1_period
            shifts[label].append(shift)
            row.append(shift)
        print(f"{v:9.0f}  " + "  ".join(f"{s:10.3f}" for s in row))
        spectra_cache.append((v, spec, results, triad1))

    # --- Figure 1: shift vs velocity ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for label in ROSSBY_LABELS:
        ax.plot(KELVIN_VELOCITIES, shifts[label], "o-", label=label)
    ax.set_xlabel("EG(1,1) (Kelvin-like mode) initial velocity (m/s)")
    ax.set_ylabel("period shift, quartet novelty − Triad 1 (days)")
    ax.set_title("Quartet C: period shift vs. Kelvin-mode driving velocity\n"
                  "(RH(4,5)/RH(3,4)/RH(1,2) held at 40m/s)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(OUT_DIR, "period_shift_vs_velocity.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print("wrote", path1)

    # --- Figure 2: spectra grid, RH(4,5), Triad 1 vs Quartet, one panel per velocity ---
    n = len(KELVIN_VELOCITIES)
    ncols = 5
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (v, spec, results, triad1) in zip(axes, spectra_cache):
        label = "RH(4,5)"
        r1, rq = results[triad1], results["full"]
        j1, jq = r1["labels"].index(label), rq["labels"].index(label)

        p1, pw1 = _power_spectrum(r1["t"], r1["E"][:, j1])
        pq, pwq = _power_spectrum(rq["t"], rq["E"][:, jq])
        m1, mq = p1 <= 20, pq <= 20
        ax.semilogy(p1[m1], pw1[m1] / pw1.max(), "b--", label="Triad 1")
        ax.semilogy(pq[mq], pwq[mq] / pwq.max(), "k-", label="Quartet")
        ax.set_xlim(0, 20)
        ax.set_ylim(1e-4, 2)
        ax.set_title(f"v_Kelvin = {v:.0f} m/s", fontsize=11)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    for ax in axes[len(spectra_cache):]:
        ax.axis("off")
    for ax in axes[::ncols]:
        ax.set_ylabel("peak-norm. power")
    for ax in axes[-ncols:]:
        ax.set_xlabel("period (days)")
    fig.suptitle("Quartet C: RH(4,5) spectrum, Triad 1 vs Quartet, across Kelvin-mode velocity", y=1.01)
    fig.tight_layout()
    path2 = os.path.join(OUT_DIR, "spectra_grid.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path2)


if __name__ == "__main__":
    main()
