import argparse
from pathlib import Path
import numpy as np

from ase.build import bulk
from ase.io import read
from icet import ClusterSpace, StructureContainer
from icet.tools import map_structure_to_reference

def snap_to_lattice(atoms, primitive, supercell):
    
    ideal=primitive.repeat(supercell)
    ideal_pos=ideal.get_positions()

    snapped=atoms.copy()
    pos=snapped.get_positions()

    for i in range(len(snapped)):
        dist=np.linalg.norm(ideal_pos-pos[i], axis=1)
        snapped.positions[i]=ideal_pos[np.argmin(dist)]

    return snapped

def traj_to_db(files, outdir, reference_file, cutoffs=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    primitive = bulk("MgO", crystalstructure="rocksalt", a=4.21)
    reference_supercell = primitive.repeat((4,4,4))
    chemical_symbols = [["Mg", "Co", "Ni", "Cu", "Zn"], ["O"]]

    if reference_file is None:
        raise ValueError("reference_file is required")

    unrelaxed = read(reference_file, index=":")
    unrelaxed_by_id = {a.info["structure_id"]: a for a in unrelaxed}

    print(f"Loaded {len(unrelaxed)} unrelaxed structures")

    for relaxed_file in files:
        mlip_name = Path(relaxed_file).stem.replace("structures_", "")
        relaxed = read(relaxed_file, index=":")

        cs = ClusterSpace(
            structure=primitive,
            cutoffs=cutoffs,
            chemical_symbols=chemical_symbols,
        )
        print(f"  ClusterSpace: {len(cs.orbit_list)} orbits")

        sc = StructureContainer(cluster_space=cs)

        n_added = 0
        n_skipped = 0

        for r in relaxed:
            sid = r.info.get("structure_id")
            if sid is None or sid not in unrelaxed_by_id:
                continue

            init_atoms = unrelaxed_by_id[sid]
            init_atoms = snap_to_lattice(init_atoms, primitive, supercell=(4,4,4))
#           mapped = map_structure_to_reference(
#               structure=init_atoms,
#               reference=reference_supercell,
#               inert_species=["O"],
#           )
            
            energy = r.info["relaxation"]["energy_per_atom_eV"]
            sc.add_structure(
                structure=init_atoms,
                properties={"energy": energy},
            )
            n_added += 1

        print(f"  Added {n_added} structures, skipped {n_skipped}")

        db_path = outdir / f"sc_{mlip_name}.tar"
        sc.write(str(db_path))
        print(f"  Saved StructureContainer: {db_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert relaxed trajectories to icet StructureContainer databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file", nargs="+", required=True,
        help="Relaxed trajectory files.",
    )
    parser.add_argument(
        "--reference", type=str, default="/home/vidit68/Desktop/hecs/data/rattled/structures_unrelaxed.traj",
        help="Unrelaxed reference trajectory.",
    )
    parser.add_argument(
        "--out", type=str, default="/home/vidit68/Desktop/hecs/data/unrattled/",
    )
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=[6.0, 4.5],
        help="Cluster cutoffs [2-body, 3-body, ...].",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    traj_to_db(
        files=args.file,
        outdir=args.out,
        reference_file=args.reference,
        cutoffs=args.cutoffs,
    )


if __name__ == "__main__":
    main()
