#!/usr/bin/env python3
"""
Generate high-entropy ceramic (HEC) structures in the rock-salt crystal structure.

Creates random solid-solution configurations by substituting cations on the
metal sublattice of a rock-salt template (e.g., MgO). Two composition modes
are supported:

  * **equimolar** (default): each element gets an equal share of the cation
    sites (±1 atom), then the assignment is shuffled randomly. This guarantees
    every structure is close to equimolar — the defining feature of a
    high-entropy system.

  * **random**: each cation site is independently assigned one of the elements
    with uniform probability. Produces a wider spread of compositions, but
    some structures may be far from equimolar.

After assignment, atomic positions are rattled slightly to break perfect
symmetry, which is important so MLIP relaxations can find genuine local
minima instead of saddle points.

Usage
-----
    # Generate 1000 equimolar structures (default)
    python structure-sim.py

    # Generate 500 random-composition structures with a fixed seed
    python structure-sim.py --n-structures 500 --mode random --seed 123

    # Custom elements and supercell
    python structure-sim.py --elements Ti Zr Hf V Nb --supercell 3 3 3
"""

import argparse
import sys
from collections import Counter

import numpy as np
from ase.build import bulk
from ase.io import write


# ---------------------------------------------------------------------------
# Composition generators
# ---------------------------------------------------------------------------

def equimolar_assignment(elements: list[str], n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a near-equimolar composition array.

    Each of the *n_elements* elements is assigned exactly ``n_sites // n_elements``
    sites. The remaining ``n_sites % n_elements`` sites are filled by randomly
    chosen elements (one extra atom each). The resulting array is then shuffled.

    Parameters
    ----------
    elements : list[str]
        Element symbols for the cation pool.
    n_sites : int
        Total number of cation sites to fill.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Shuffled array of element symbols, length *n_sites*.
    """
    n_elem = len(elements)
    base_count = n_sites // n_elem
    remainder = n_sites % n_elem

    # Build the assignment: base_count of each, plus one extra for 'remainder' elements
    assignment = []
    extras = rng.choice(elements, size=remainder, replace=False)
    for elem in elements:
        count = base_count + (1 if elem in extras else 0)
        assignment.extend([elem] * count)

    assignment = np.array(assignment)
    rng.shuffle(assignment)
    return assignment


def random_assignment(elements: list[str], n_sites: int, rng: np.random.Generator) -> np.ndarray:
    """
    Assign each cation site independently and uniformly from the element pool.

    Parameters
    ----------
    elements : list[str]
        Element symbols for the cation pool.
    n_sites : int
        Total number of cation sites to fill.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of element symbols, length *n_sites*.
    """
    return rng.choice(elements, size=n_sites, replace=True)


# ---------------------------------------------------------------------------
# Structure builder
# ---------------------------------------------------------------------------

def generate_hec_structures(
    elements: list[str],
    n_structures: int = 1000,
    supercell: tuple[int, int, int] = (4, 4, 4),
    lattice_constant: float = 4.21,
    mode: str = "equimolar",
    rattle_stdev: float = 0.05,
    seed: int = 42,
) -> list:
    """
    Generate a list of high-entropy ceramic (rock-salt) ASE Atoms objects.

    Parameters
    ----------
    elements : list[str]
        Cation element symbols (e.g. ['Mg', 'Co', 'Ni', 'Cu', 'Zn']).
    n_structures : int
        Number of structures to generate.
    supercell : tuple[int, int, int]
        Supercell dimensions applied to the primitive rock-salt cell.
    lattice_constant : float
        Lattice constant (Å) for the rock-salt template.
    mode : {'equimolar', 'random'}
        Composition assignment strategy.
    rattle_stdev : float
        Standard deviation (Å) for Gaussian displacement of atoms.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[ase.Atoms]
        Generated structures with metadata stored in ``atoms.info``.
    """
    rng = np.random.default_rng(seed)

    assign_fn = equimolar_assignment if mode == "equimolar" else random_assignment

    # Build the rock-salt template from MgO
    template = bulk("MgO", crystalstructure="rocksalt", a=lattice_constant)
    template = template.repeat(supercell)

    # Identify cation (Mg) site indices
    metal_idx = [i for i, s in enumerate(template.get_chemical_symbols()) if s == "Mg"]
    n_sites = len(metal_idx)

    print(f"Template: {len(template)} atoms ({n_sites} cation sites, "
          f"{len(template) - n_sites} anion sites)")
    print(f"Elements: {elements}")
    print(f"Mode: {mode} | Supercell: {supercell} | Rattle σ: {rattle_stdev} Å")
    print(f"Generating {n_structures} structures (seed={seed}) ...")

    structures = []
    composition_counts = Counter()  # track overall composition statistics

    for i in range(n_structures):
        atoms = template.copy()
        symbols = list(atoms.get_chemical_symbols())

        # Assign cations
        metals = assign_fn(elements, n_sites, rng)
        for idx, elem in zip(metal_idx, metals):
            symbols[idx] = elem
        atoms.set_chemical_symbols(symbols)

        # Rattle to break symmetry
        atoms.rattle(stdev=rattle_stdev, rng=rng)

        # Store per-structure metadata
        comp = dict(Counter(metals))
        fractions = {k: v / n_sites for k, v in comp.items()}
        atoms.info["structure_id"] = i
        atoms.info["composition"] = comp
        atoms.info["composition_fractions"] = fractions
        atoms.info["mode"] = mode
        atoms.info["seed"] = seed
        atoms.info["n_cation_sites"] = n_sites

        structures.append(atoms)
        composition_counts.update(comp)

    # Print composition statistics
    print(f"\n{'─' * 50}")
    print("Composition statistics (averaged over all structures):")
    total_assignments = n_structures * n_sites
    for elem in sorted(elements):
        count = composition_counts.get(elem, 0)
        frac = count / total_assignments
        print(f"  {elem:>3s}: {frac:.4f}  ({count}/{total_assignments})")
    ideal = 1.0 / len(elements)
    print(f"  Ideal equimolar fraction: {ideal:.4f}")
    print(f"{'─' * 50}")

    return structures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate high-entropy ceramic (rock-salt) structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--elements", nargs="+", default=["Mg", "Co", "Ni", "Cu", "Zn"],
        help="Cation elements for the HEC.",
    )
    parser.add_argument(
        "--n-structures", type=int, default=1000,
        help="Number of structures to generate.",
    )
    parser.add_argument(
        "--supercell", type=int, nargs=3, default=[4, 4, 4],
        help="Supercell dimensions (nx ny nz).",
    )
    parser.add_argument(
        "--lattice-constant", type=float, default=4.21,
        help="Rock-salt lattice constant in Å (default is MgO's value).",
    )
    parser.add_argument(
        "--mode", choices=["equimolar", "random"], default="equimolar",
        help="Composition assignment mode.",
    )
    parser.add_argument(
        "--rattle-stdev", type=float, default=0.05,
        help="Rattle standard deviation in Å.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output", type=str, default="structures_unrelaxed.traj",
        help="Output trajectory file.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    structures = generate_hec_structures(
        elements=args.elements,
        n_structures=args.n_structures,
        supercell=tuple(args.supercell),
        lattice_constant=args.lattice_constant,
        mode=args.mode,
        rattle_stdev=args.rattle_stdev,
        seed=args.seed,
    )

    write(args.output, structures)
    print(f"\n✓ Wrote {len(structures)} structures to '{args.output}'")


if __name__ == "__main__":
    main()
