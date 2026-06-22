import os
import argparse
import glob
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

def compute_sro(atoms, cutoff=3.0):

    elements=np.array(atoms.get_chemical_symbols())
    t_atoms=len(elements)
    u_elements=sorted(list(set(elements)))

    concentrations={}
    for e in u_elements:
        concentrations[e]=np.sum(elements==e)/t_atoms

    #Neighborhood
    i, j = neighbor_list('ij', atoms, cutoff)
    elements_i=elements[i]
    elements_j=elements[j]

    #Matrix calculations
    sro_matrix={e: {} for e in u_elements}

    for e_a in u_elements:
        a_bonds=(elements_i==e_a)
        total_a=np.sum(a_bonds)

        if total_a==0:
            for e_b in u_elements:
                sro_matrix[e_a][e_b]=0.0
            continue

        for e_b in u_elements:
            c_b=concentrations[e_b]

            a_b_bonds=np.sum(a_bonds & (elements_j==e_b))
            P_ab=a_b_bonds/total_a

            alpha=1.0-(P_ab/c_b)

            sro_matrix[e_a][e_b]=alpha
    return sro_matrix, u_elements

def print_sro_matrix(matrix, elements):
    header = "    " + "".join([f"{e:>8}" for e in elements])
    print(header)
    for e_a in elements:
        row=f"{e_a:<4}"
        for e_b in elements:
            val=matrix[e_a][e_b]
            row+=f"{val:>8.4f}"
        print(row)
    

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

    ref_structures = None
    if reference_file:
        print(f"\nLoading reference (unrelaxed baseline): {reference_file}")
        ref_structures = read(reference_file, index=":")
        print(f"  → {len(ref_structures)} structures")

    print("COMPUTING METRICS")

    all_sro={}
    if ref_structures:
        ref_atoms = ref_structures[0]
        ref_matrix, u_elems = compute_sro(ref_atoms)
        all_sro["Reference"] = ref_matrix
        
        print("\n[Reference] Unrelaxed Initial SRO Matrix:")
        print_sro_matrix(ref_matrix, u_elems)

    for label, structures in all_structures.items():
        final_atoms = structures[-1]
        matrix, u_elems = compute_sro(final_atoms)
        all_sro[label] = matrix
        
        print(f"\n[{label}] Final Relaxed SRO Matrix:")
        print_sro_matrix(matrix, u_elems)
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
