#!/usr/bin/env python3
"""
Relax high-entropy ceramic structures using Machine Learning Interatomic
Potentials (MLIPs) through the ASE calculator interface.

Supported calculators
---------------------
- **chgnet**  : CHGNet (Deng et al., Nature Machine Intelligence 2023).
                Pretrained universal potential with charge awareness.
- **mace**    : MACE-MP-0 foundation model (Batatia et al., 2024).
                Higher-order equivariant message passing.
- **eqv2**    : EquiformerV2 / EScAIP via fairchem-core (Meta/FAIR).
                Transformer-based, trained on OC20/OC22/OMAT datasets.

Each calculator is loaded via a factory function and attached to an ASE
``Atoms`` object.  Relaxation uses LBFGS (default) or FIRE, optionally
wrapping the atoms in ``FrechetCellFilter`` to simultaneously optimise
cell shape/volume *and* atomic positions.

Usage
-----
    # Relax all structures with CHGNet
    python relax_structures.py --calculator chgnet \\
        --input structures_unrelaxed.traj --output structures_chgnet.traj

    # Relax with MACE, position-only, tighter convergence
    python relax_structures.py --calculator mace --fmax 0.01 --no-cell-relax \\
        --input structures_unrelaxed.traj --output structures_mace.traj

    # Relax a subset (first 10 structures) with EquiformerV2
    python relax_structures.py --calculator eqv2 --max-structures 10 \\
        --input structures_unrelaxed.traj --output structures_eqv2.traj
"""

import argparse
import sys
import time
import traceback

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS, FIRE

# FrechetCellFilter is preferred over ExpCellFilter (better performance).
# Fall back to ExpCellFilter for older ASE versions.
try:
    from ase.filters import FrechetCellFilter as CellFilter
except ImportError:
    from ase.filters import ExpCellFilter as CellFilter


# ---------------------------------------------------------------------------
# Calculator factory
# ---------------------------------------------------------------------------

def get_calculator(name: str, device: str = "auto"):
    """
    Return an ASE calculator for the requested MLIP.

    Parameters
    ----------
    name : str
        One of 'chgnet', 'mace', 'eqv2'.
    device : str
        Device string ('cpu', 'cuda', 'auto'). 'auto' uses CUDA if available.

    Returns
    -------
    ase.calculators.calculator.Calculator
        An ASE-compatible calculator.

    Raises
    ------
    ImportError
        If the required package is not installed.
    ValueError
        If *name* is not recognised.
    """
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    name = name.lower()

    if name == "chgnet":
        from chgnet.model.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        model = CHGNet.load()
        calc = CHGNetCalculator(model=model, use_device=device)
        print(f"  Loaded CHGNet (device={device})")
        return calc

    elif name == "mace":
        from mace.calculators import mace_mp

        # MACE-MP-0 foundation model; "medium" is a good accuracy/speed balance
        calc = mace_mp(model="medium", device=device, default_dtype="float64")
        print(f"  Loaded MACE-MP-0 medium (device={device})")
        return calc

    elif name == "eqv2":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        # UMA (Universal Materials Architecture) model — use "omat" task for bulk materials
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=device)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        print(f"  Loaded FAIRChem UMA-s (device={device}, task=omat)")
        return calc

    else:
        raise ValueError(
            f"Unknown calculator '{name}'. Choose from: chgnet, mace, eqv2"
        )


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

def relax_structure(
    atoms,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: str = "LBFGS",
    relax_cell: bool = True,
) -> dict:
    """
    Relax a single ASE Atoms object.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure with a calculator already attached.
    fmax : float
        Maximum force convergence criterion (eV/Å).
    max_steps : int
        Maximum optimisation steps.
    optimizer : str
        'LBFGS' or 'FIRE'.
    relax_cell : bool
        If True, simultaneously optimise cell shape/volume using FrechetCellFilter.

    Returns
    -------
    dict
        Relaxation results: converged, energy, max_force, volume, n_steps, time.
    """
    t0 = time.perf_counter()

    # Wrap in cell filter if requested
    opt_target = CellFilter(atoms) if relax_cell else atoms

    # Choose optimizer
    OptClass = LBFGS if optimizer == "LBFGS" else FIRE
    opt = OptClass(opt_target, logfile=None)  # suppress per-step output

    converged = opt.run(fmax=fmax, steps=max_steps)

    # Gather results
    forces = atoms.get_forces()
    max_force = float(np.max(np.linalg.norm(forces, axis=1)))
    energy = float(atoms.get_potential_energy())
    volume = float(atoms.get_volume())
    elapsed = time.perf_counter() - t0

    return {
        "converged": converged,
        "energy_eV": energy,
        "energy_per_atom_eV": energy / len(atoms),
        "max_force_eV_A": max_force,
        "volume_A3": volume,
        "volume_per_atom_A3": volume / len(atoms),
        "n_steps": opt.nsteps,
        "time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_relaxation(
    input_file: str,
    output_file: str,
    calculator_name: str,
    fmax: float = 0.05,
    max_steps: int = 200,
    optimizer: str = "LBFGS",
    relax_cell: bool = True,
    device: str = "auto",
    max_structures: int | None = None,
):
    """
    Load structures, relax with an MLIP, and save results.

    Parameters
    ----------
    input_file : str
        Path to input trajectory/file.
    output_file : str
        Path to output trajectory.
    calculator_name : str
        MLIP name ('chgnet', 'mace', 'eqv2').
    fmax : float
        Force convergence criterion.
    max_steps : int
        Maximum optimiser steps per structure.
    optimizer : str
        'LBFGS' or 'FIRE'.
    relax_cell : bool
        Whether to relax cell parameters.
    device : str
        Device for the MLIP.
    max_structures : int or None
        If set, only relax the first N structures.
    """
    # Load structures
    print(f"Loading structures from '{input_file}' ...")
    structures = read(input_file, index=":")
    if max_structures is not None:
        structures = structures[:max_structures]
    print(f"  {len(structures)} structures loaded")

    # Load calculator
    print(f"\nInitialising calculator: {calculator_name}")
    calc = get_calculator(calculator_name, device=device)

    # Relaxation loop
    print(f"\nRelaxing {len(structures)} structures "
          f"(fmax={fmax}, max_steps={max_steps}, "
          f"optimizer={optimizer}, cell_relax={relax_cell})")
    print(f"{'─' * 70}")

    relaxed = []
    n_converged = 0
    n_failed = 0
    total_t0 = time.perf_counter()

    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(structures), total=len(structures), desc="Relaxing")
    except ImportError:
        iterator = enumerate(structures)
        print("  (install tqdm for progress bars: pip install tqdm)")

    for i, atoms in iterator:
        try:
            # Attach calculator (each structure needs its own calc attachment)
            atoms.calc = calc

            result = relax_structure(
                atoms,
                fmax=fmax,
                max_steps=max_steps,
                optimizer=optimizer,
                relax_cell=relax_cell,
            )

            # Store metadata
            atoms.info["calculator"] = calculator_name
            atoms.info["relaxation"] = result
            atoms.info["relaxed"] = True

            relaxed.append(atoms)

            if result["converged"]:
                n_converged += 1

            # Print occasional progress (every 100 structures if no tqdm)
            if not isinstance(iterator, enumerate) or (i + 1) % 100 == 0:
                if isinstance(iterator, enumerate):
                    print(f"  [{i+1}/{len(structures)}] "
                          f"E={result['energy_per_atom_eV']:.4f} eV/atom, "
                          f"fmax={result['max_force_eV_A']:.4f}, "
                          f"steps={result['n_steps']}, "
                          f"{'✓' if result['converged'] else '✗'}")

        except Exception as e:
            n_failed += 1
            print(f"\n  ⚠ Structure {i} FAILED: {e}")
            traceback.print_exc()
            # Still append with failure metadata
            atoms.info["relaxed"] = False
            atoms.info["error"] = str(e)
            relaxed.append(atoms)

    total_time = time.perf_counter() - total_t0

    # Save
    write(output_file, relaxed)

    # Summary
    print(f"\n{'═' * 70}")
    print(f"RELAXATION SUMMARY ({calculator_name})")
    print(f"{'─' * 70}")
    print(f"  Total structures : {len(structures)}")
    print(f"  Converged        : {n_converged} ({100*n_converged/len(structures):.1f}%)")
    print(f"  Failed           : {n_failed}")
    print(f"  Total time       : {total_time:.1f} s ({total_time/len(structures):.2f} s/structure)")
    print(f"  Output           : {output_file}")
    print(f"{'═' * 70}")

    # Quick energy statistics
    energies = [a.info["relaxation"]["energy_per_atom_eV"]
                for a in relaxed if a.info.get("relaxed")]
    if energies:
        print(f"\n  Energy/atom (eV): mean={np.mean(energies):.4f}, "
              f"std={np.std(energies):.4f}, "
              f"min={np.min(energies):.4f}, max={np.max(energies):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Relax HEC structures with an MLIP calculator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--calculator", type=str, required=True,
        choices=["chgnet", "mace", "eqv2"],
        help="MLIP calculator to use.",
    )
    parser.add_argument(
        "--input", type=str, default="structures_unrelaxed.traj",
        help="Input trajectory file.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output trajectory file (default: structures_<calculator>.traj).",
    )
    parser.add_argument(
        "--fmax", type=float, default=0.05,
        help="Force convergence criterion (eV/Å).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum optimiser steps per structure.",
    )
    parser.add_argument(
        "--optimizer", choices=["LBFGS", "FIRE"], default="LBFGS",
        help="ASE optimizer to use.",
    )
    parser.add_argument(
        "--no-cell-relax", action="store_true",
        help="Disable cell relaxation (only relax positions).",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for the MLIP model.",
    )
    parser.add_argument(
        "--max-structures", type=int, default=None,
        help="Only relax the first N structures (for testing).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    output = args.output or f"structures_{args.calculator}.traj"

    run_relaxation(
        input_file=args.input,
        output_file=output,
        calculator_name=args.calculator,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        relax_cell=not args.no_cell_relax,
        device=args.device,
        max_structures=args.max_structures,
    )


if __name__ == "__main__":
    main()
