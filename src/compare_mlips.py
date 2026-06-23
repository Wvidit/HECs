import os
import argparse
import glob
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
import matplotlib.pyplot as plt

def snap_to_lattice(atoms, primitive, supercell):
    
    ideal=primitive.repeat(supercell)
    ideal_pos=ideal.get_positions()

    snapped=atoms.copy()
    pos=snapped.get_positions()

    for i in range(len(snapped)):
        dist=np.linalg.norm(ideal_pos-pos[i], axis=1)
        snapped.positions[i]=ideal_pos[np.argmin(dist)]

    return snapped

def print_sro_matrix(matrix, elements):
    header = "    " + "".join([f"{e:>8}" for e in elements])
    print(header)
    for e_a in elements:
        row=f"{e_a:<4}"
        for e_b in elements:
            val=matrix[e_a][e_b]
            row+=f"{val:>8.4f}"
        print(row)

def save_sro_matrix_plot(sro_matrix, elements, save_path="sro_matrix.png"):
    """
    Save SRO matrix as a heatmap image.
    """
    n = len(elements)
    matrix = np.zeros((n, n))
    for i, e_a in enumerate(elements):
        for j, e_b in enumerate(elements):
            matrix[i, j] = sro_matrix[e_a][e_b]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(elements)
    ax.set_yticklabels(elements)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title("Short-Range Order (SRO) Matrix")
    ax.set_xlabel("Neighbor Element")
    ax.set_ylabel("Central Element")

    plt.colorbar(im, ax=ax, label="SRO Parameter α")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SRO matrix plot to {save_path}")

def compute_sro(atoms, cutoff, save_path):

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

    save_sro_matrix_plot(sro_matrix, u_elements, save_path)
    return sro_matrix, u_elements

def compute_sro_icet(atoms, cutoff, save_path):
    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    from icet import ClusterSpace

    primitive = bulk("MgO", crystalstructure="rocksalt", a=4.21)
    cs = ClusterSpace(                                      # validates the system
        structure=primitive,
        cutoffs=[cutoff],
        chemical_symbols=[["Mg", "Co", "Ni", "Cu", "Zn"], ["O"]]
    )

    # supercell dims from cell lengths
    prim_lengths = primitive.cell.lengths()
    atoms_lengths = atoms.cell.lengths()
    supercell = tuple(int(round(a / p)) for a, p in zip(atoms_lengths, prim_lengths))
    snapped = snap_to_lattice(atoms, primitive, supercell)

    # extracting cations
    cation_indices = np.where((np.array(snapped.get_chemical_symbols()) != 'O'))[0]
    snapped = snapped[cation_indices]

    elements   = np.array(snapped.get_chemical_symbols())
    u_elements = sorted(set(elements))
    total      = len(elements)
    conc       = {e: np.sum(elements == e) / total for e in u_elements}

    i_arr, j_arr = neighbor_list('ij', snapped, cutoff)
    elems_i = elements[i_arr]
    elems_j = elements[j_arr]

    sro_matrix = {e_a: {} for e_a in u_elements}
    for e_a in u_elements:
        mask_a  = (elems_i == e_a)
        total_a = np.sum(mask_a)
        for e_b in u_elements:
            if total_a == 0 or conc[e_b] == 0:
                sro_matrix[e_a][e_b] = 0.0
                continue
            n_ab = np.sum(mask_a & (elems_j == e_b))
            sro_matrix[e_a][e_b] = 1.0 - (n_ab / total_a) / conc[e_b]

    save_sro_matrix_plot(sro_matrix, u_elements, save_path)
    return sro_matrix, u_elements
    

def run_comparison(
    files: list[str],
    labels: list[str],
    outdir: str,
    reference_file: str,
    save: str
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
        ref_matrix, u_elems = compute_sro_icet(ref_atoms, cutoff=3, save_path="/home/vidit68/Desktop/hecs/results/sro_initial_icet.jpg")
        all_sro["Reference"] = ref_matrix
        
        print("\n[Reference] Unrelaxed Initial SRO Matrix:")
        print_sro_matrix(ref_matrix, u_elems)

    for label, structures in all_structures.items():
        final_atoms = structures[-1]
        matrix, u_elems = compute_sro_icet(final_atoms, cutoff=3, save_path=save)
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
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the sro matrices",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_comparison(
        files=args.files,
        labels=args.labels,
        outdir=args.outdir,
        reference_file=args.reference,
        save=args.save
    )


if __name__ == "__main__":
    main()
