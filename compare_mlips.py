#!/usr/bin/env python3
"""
Compare structures relaxed by different MLIPs.

Reads two or more trajectory files (each from a different MLIP) and computes
pairwise comparison metrics:

  * **Energy per atom** — scatter plots and histograms
  * **Volume per atom** — how much each MLIP relaxes the cell
  * **Maximum residual force** — convergence quality
  * **Structural RMSD** — atomic position deviation between MLIPs
  * **Lattice parameters** — cell shape changes (a, b, c, α, β, γ)

Generates publication-quality matplotlib figures saved as PNGs.

Usage
-----
    # Compare CHGNet vs MACE
    python compare_mlips.py \\
        --files structures_chgnet.traj structures_mace.traj \\
        --labels CHGNet MACE

    # Compare all three with custom output directory
    python compare_mlips.py \\
        --files structures_chgnet.traj structures_mace.traj structures_eqv2.traj \\
        --labels CHGNet MACE EqV2 \\
        --outdir comparison_plots

    # Also include unrelaxed structures as a reference baseline
    python compare_mlips.py \\
        --files structures_chgnet.traj structures_mace.traj \\
        --labels CHGNet MACE \\
        --reference structures_unrelaxed.traj
"""

import argparse
import os
import sys

import numpy as np
from ase.io import read


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_energies_per_atom(structures: list) -> np.ndarray:
    """Extract energy per atom from relaxed structures."""
    energies = []
    for atoms in structures:
        # Try to get energy from info dict (stored during relaxation)
        if "relaxation" in atoms.info:
            energies.append(atoms.info["relaxation"]["energy_per_atom_eV"])
        elif atoms.calc is not None:
            try:
                energies.append(atoms.get_potential_energy() / len(atoms))
            except Exception:
                energies.append(np.nan)
        else:
            energies.append(np.nan)
    return np.array(energies)


def get_volumes_per_atom(structures: list) -> np.ndarray:
    """Extract volume per atom."""
    return np.array([atoms.get_volume() / len(atoms) for atoms in structures])


def get_max_forces(structures: list) -> np.ndarray:
    """Extract maximum residual force magnitude from relaxed structures."""
    forces = []
    for atoms in structures:
        if "relaxation" in atoms.info:
            forces.append(atoms.info["relaxation"]["max_force_eV_A"])
        elif atoms.calc is not None:
            try:
                f = atoms.get_forces()
                forces.append(float(np.max(np.linalg.norm(f, axis=1))))
            except Exception:
                forces.append(np.nan)
        else:
            forces.append(np.nan)
    return np.array(forces)


def get_lattice_params(structures: list) -> np.ndarray:
    """Extract lattice parameters (a, b, c, α, β, γ) for each structure."""
    return np.array([atoms.cell.cellpar() for atoms in structures])


def compute_rmsd(structures_a: list, structures_b: list) -> np.ndarray:
    """
    Compute per-structure RMSD of atomic positions between two trajectory sets.

    Both sets must have the same number of structures and same atom ordering.
    Positions are compared directly (no alignment), which is appropriate when
    structures share the same initial configuration.

    Parameters
    ----------
    structures_a, structures_b : list[ase.Atoms]
        Two sets of structures to compare.

    Returns
    -------
    np.ndarray
        RMSD values (Å), one per structure pair.
    """
    assert len(structures_a) == len(structures_b), \
        f"Trajectory lengths differ: {len(structures_a)} vs {len(structures_b)}"

    rmsds = []
    for a, b in zip(structures_a, structures_b):
        disp = a.get_positions() - b.get_positions()
        rmsd = np.sqrt(np.mean(np.sum(disp**2, axis=1)))
        rmsds.append(rmsd)
    return np.array(rmsds)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.figsize": (10, 7),
        "figure.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
    })
    return plt


def plot_energy_comparison(all_energies: dict, outdir: str):
    """Scatter plot and histogram of energy per atom across MLIPs."""
    plt = setup_matplotlib()
    labels = list(all_energies.keys())
    n = len(labels)

    # --- Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    for label, color in zip(labels, colors):
        e = all_energies[label]
        valid = e[~np.isnan(e)]
        ax.hist(valid, bins=50, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Energy per atom (eV)")
    ax.set_ylabel("Count")
    ax.set_title("Energy Distribution by MLIP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "energy_histogram.png"))
    plt.close(fig)
    print(f"  Saved energy_histogram.png")

    # --- Pairwise scatter (for 2+ MLIPs) ---
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                fig, ax = plt.subplots(figsize=(8, 8))
                ei = all_energies[labels[i]]
                ej = all_energies[labels[j]]
                valid = ~(np.isnan(ei) | np.isnan(ej))
                ax.scatter(ei[valid], ej[valid], alpha=0.4, s=10, color="steelblue")

                # Perfect agreement line
                emin = min(ei[valid].min(), ej[valid].min())
                emax = max(ei[valid].max(), ej[valid].max())
                margin = (emax - emin) * 0.05
                lims = [emin - margin, emax + margin]
                ax.plot(lims, lims, "k--", alpha=0.5, label="y = x")
                ax.set_xlim(lims)
                ax.set_ylim(lims)

                # Statistics
                diff = ej[valid] - ei[valid]
                mae = np.mean(np.abs(diff))
                rmse = np.sqrt(np.mean(diff**2))
                ax.text(0.05, 0.92, f"MAE = {mae:.4f} eV/atom\nRMSE = {rmse:.4f} eV/atom",
                        transform=ax.transAxes, fontsize=11,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

                ax.set_xlabel(f"{labels[i]} energy/atom (eV)")
                ax.set_ylabel(f"{labels[j]} energy/atom (eV)")
                ax.set_title(f"Energy Correlation: {labels[i]} vs {labels[j]}")
                ax.set_aspect("equal")
                ax.legend()
                fig.tight_layout()
                fname = f"energy_scatter_{labels[i]}_vs_{labels[j]}.png"
                fig.savefig(os.path.join(outdir, fname))
                plt.close(fig)
                print(f"  Saved {fname}")


def plot_volume_comparison(all_volumes: dict, outdir: str):
    """Histogram of volume per atom across MLIPs."""
    plt = setup_matplotlib()
    labels = list(all_volumes.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in zip(labels, colors):
        v = all_volumes[label]
        ax.hist(v, bins=50, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Volume per atom (ų)")
    ax.set_ylabel("Count")
    ax.set_title("Volume Distribution by MLIP")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "volume_histogram.png"))
    plt.close(fig)
    print(f"  Saved volume_histogram.png")


def plot_force_comparison(all_forces: dict, outdir: str):
    """Histogram of max residual forces (convergence quality)."""
    plt = setup_matplotlib()
    labels = list(all_forces.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in zip(labels, colors):
        f = all_forces[label]
        valid = f[~np.isnan(f)]
        ax.hist(valid, bins=50, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_xlabel("Max residual force (eV/Å)")
    ax.set_ylabel("Count")
    ax.set_title("Convergence Quality: Max Force Distribution")
    ax.axvline(0.05, color="red", linestyle="--", alpha=0.5, label="fmax = 0.05")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "force_histogram.png"))
    plt.close(fig)
    print(f"  Saved force_histogram.png")


def plot_rmsd_comparison(rmsd_data: dict, outdir: str):
    """Histogram of RMSD between MLIP pairs."""
    plt = setup_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(rmsd_data)))
    for (pair_label, rmsds), color in zip(rmsd_data.items(), colors):
        ax.hist(rmsds, bins=50, alpha=0.6, label=pair_label, color=color, edgecolor="white")
    ax.set_xlabel("RMSD (Å)")
    ax.set_ylabel("Count")
    ax.set_title("Structural RMSD Between MLIPs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rmsd_histogram.png"))
    plt.close(fig)
    print(f"  Saved rmsd_histogram.png")


def plot_lattice_comparison(all_lattice: dict, outdir: str):
    """Box plots of lattice parameters across MLIPs."""
    plt = setup_matplotlib()
    labels = list(all_lattice.keys())
    param_names = ["a (Å)", "b (Å)", "c (Å)", "α (°)", "β (°)", "γ (°)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, (ax, pname) in enumerate(zip(axes.flat, param_names)):
        data = [all_lattice[label][:, idx] for label in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(pname)
        ax.set_title(pname)

    fig.suptitle("Lattice Parameter Distributions by MLIP", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "lattice_params.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved lattice_params.png")


def plot_summary_table(all_energies, all_volumes, all_forces, outdir):
    """Generate a summary statistics table as an image."""
    plt = setup_matplotlib()

    labels = list(all_energies.keys())
    rows = []
    for label in labels:
        e = all_energies[label]
        v = all_volumes[label]
        f = all_forces[label]
        valid_e = e[~np.isnan(e)]
        valid_f = f[~np.isnan(f)]
        rows.append([
            label,
            f"{np.mean(valid_e):.4f} ± {np.std(valid_e):.4f}",
            f"{np.mean(v):.3f} ± {np.std(v):.3f}",
            f"{np.mean(valid_f):.4f}",
            f"{np.sum(valid_f <= 0.05)} / {len(valid_f)}",
        ])

    fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(labels)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["MLIP", "E/atom (eV)", "V/atom (ų)", "Mean fmax (eV/Å)", "Converged"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    fig.suptitle("MLIP Comparison Summary", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "summary_table.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary_table.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(
    files: list[str],
    labels: list[str],
    outdir: str = "comparison_plots",
    reference_file: str | None = None,
):
    """
    Run the full comparison pipeline.

    Parameters
    ----------
    files : list[str]
        Paths to trajectory files from different MLIPs.
    labels : list[str]
        Human-readable labels for each MLIP.
    outdir : str
        Output directory for plots.
    reference_file : str or None
        Optional unrelaxed reference trajectory for baseline comparison.
    """
    assert len(files) == len(labels), \
        f"Number of files ({len(files)}) must match labels ({len(labels)})"

    os.makedirs(outdir, exist_ok=True)

    # Load all trajectories
    all_structures = {}
    for filepath, label in zip(files, labels):
        print(f"Loading {label}: {filepath}")
        structures = read(filepath, index=":")
        all_structures[label] = structures
        print(f"  → {len(structures)} structures")

    # Verify all have the same length
    lengths = {label: len(s) for label, s in all_structures.items()}
    if len(set(lengths.values())) > 1:
        print(f"\n⚠ Warning: trajectory lengths differ: {lengths}")
        min_len = min(lengths.values())
        print(f"  Truncating all to {min_len} structures for comparison")
        all_structures = {k: v[:min_len] for k, v in all_structures.items()}

    # Load reference if provided
    if reference_file:
        print(f"\nLoading reference: {reference_file}")
        ref_structures = read(reference_file, index=":")
        print(f"  → {len(ref_structures)} structures")

    # --- Compute metrics ---
    print(f"\n{'═' * 60}")
    print("COMPUTING METRICS")
    print(f"{'═' * 60}")

    all_energies = {}
    all_volumes = {}
    all_forces = {}
    all_lattice = {}

    for label, structures in all_structures.items():
        print(f"\n{label}:")
        e = get_energies_per_atom(structures)
        v = get_volumes_per_atom(structures)
        f = get_max_forces(structures)
        lp = get_lattice_params(structures)

        all_energies[label] = e
        all_volumes[label] = v
        all_forces[label] = f
        all_lattice[label] = lp

        valid_e = e[~np.isnan(e)]
        valid_f = f[~np.isnan(f)]
        print(f"  Energy/atom: {np.mean(valid_e):.4f} ± {np.std(valid_e):.4f} eV")
        print(f"  Volume/atom: {np.mean(v):.3f} ± {np.std(v):.3f} ų")
        print(f"  Max force  : {np.mean(valid_f):.4f} ± {np.std(valid_f):.4f} eV/Å")
        print(f"  Converged  : {np.sum(valid_f <= 0.05)}/{len(valid_f)} "
              f"({100*np.sum(valid_f <= 0.05)/len(valid_f):.1f}%)")

    # Pairwise RMSD
    rmsd_data = {}
    labels_list = list(all_structures.keys())
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            li, lj = labels_list[i], labels_list[j]
            try:
                rmsds = compute_rmsd(all_structures[li], all_structures[lj])
                pair_label = f"{li} vs {lj}"
                rmsd_data[pair_label] = rmsds
                print(f"\n  RMSD ({pair_label}): {np.mean(rmsds):.4f} ± {np.std(rmsds):.4f} Å")
            except AssertionError as e:
                print(f"\n  ⚠ Cannot compute RMSD for {li} vs {lj}: {e}")

    # Pairwise energy differences
    print(f"\n{'─' * 60}")
    print("PAIRWISE ENERGY DIFFERENCES (eV/atom)")
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            li, lj = labels_list[i], labels_list[j]
            ei, ej = all_energies[li], all_energies[lj]
            valid = ~(np.isnan(ei) | np.isnan(ej))
            diff = ej[valid] - ei[valid]
            print(f"  {lj} − {li}: mean={np.mean(diff):.4f}, "
                  f"std={np.std(diff):.4f}, "
                  f"MAE={np.mean(np.abs(diff)):.4f}")

    # --- Generate plots ---
    print(f"\n{'═' * 60}")
    print(f"GENERATING PLOTS → {outdir}/")
    print(f"{'═' * 60}")

    plot_energy_comparison(all_energies, outdir)
    plot_volume_comparison(all_volumes, outdir)
    plot_force_comparison(all_forces, outdir)
    if rmsd_data:
        plot_rmsd_comparison(rmsd_data, outdir)
    plot_lattice_comparison(all_lattice, outdir)
    plot_summary_table(all_energies, all_volumes, all_forces, outdir)

    print(f"\n✓ All plots saved to '{outdir}/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare structures relaxed by different MLIPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="Trajectory files to compare (one per MLIP).",
    )
    parser.add_argument(
        "--labels", nargs="+", required=True,
        help="Labels for each trajectory (must match --files count).",
    )
    parser.add_argument(
        "--outdir", type=str, default="comparison_plots",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Optional unrelaxed reference trajectory.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_comparison(
        files=args.files,
        labels=args.labels,
        outdir=args.outdir,
        reference_file=args.reference,
    )


if __name__ == "__main__":
    main()
