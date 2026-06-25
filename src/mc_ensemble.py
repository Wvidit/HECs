# Usage 
# python src/mc_ensemble.py --ce_src results/clusterexpansions/uma.tar --save results/

import argparse
import numpy as np
from ase.neighborlist import neighbor_list
from mchammer.observers.base_observer import BaseObserver

class MulticomponentSROObserver(BaseObserver):
    """
    Custom mchammer observer to track Warren-Cowley SRO for multicomponent ceramics.
    Calculates SRO exclusively for the cation sublattice.
    """
    def __init__(self, interval: int, cutoff: float, elements: list[str]):
        super().__init__(interval=interval, return_type=dict, tag='MulticomponentSRO')
        self.cutoff = cutoff
        self.elements = sorted(elements)

    def get_observable(self, structure) -> dict:
        cation_indices = np.where((np.array(structure.get_chemical_symbols()) != 'O'))[0]
        cations_only = structure[cation_indices]

        elements = np.array(cations_only.get_chemical_symbols())
        total = len(elements)
        conc = {e: np.sum(elements == e) / total for e in self.elements}

        i_arr, j_arr = neighbor_list('ij', cations_only, self.cutoff)
        elems_i = elements[i_arr]
        elems_j = elements[j_arr]

        sro_dict = {}
        for e_a in self.elements:
            mask_a = (elems_i == e_a)
            total_a = np.sum(mask_a)

            for e_b in self.elements:
                key = f"SRO_{e_a}_{e_b}"
                if total_a == 0 or conc[e_b] == 0:
                    sro_dict[key] = 0.0
                    continue

                n_ab = np.sum(mask_a & (elems_j == e_b))
                sro_dict[key] = 1.0 - (n_ab / total_a) / conc[e_b]

        return sro_dict


def mc_ensembler(path, save, temperatures=[500, 750, 1000, 1250, 1500, 1750, 2000]):
    from icet import ClusterExpansion
    from ase.build import make_supercell
    from mchammer.calculators import ClusterExpansionCalculator
    from mchammer.ensembles import CanonicalEnsemble

    ce_loaded = ClusterExpansion.read(path)
    
    cations=["Mg", "Co", "Ni", "Cu", "Zn"]
    seed = 42 
    rng = np.random.default_rng(seed)

    primitive = ce_loaded._cluster_space.primitive_structure.copy()
    supercell = make_supercell(primitive, [[-1,1,1],[1,-1,1],[1,1,-1]]) # this convertes fcc->cubic
    supercell = supercell.repeat(4) # 4x4x4

    cations_indices = [atom.index for atom in supercell if atom.symbol != 'O']
    n_sites = len(cations_indices)

    # making equimolarity
    base_count = n_sites // 5
    remainder = n_sites % 5

    assignment = []
    extras = rng.choice(cations, size=remainder, replace=False)
    for elem in cations:
        count = base_count + (1 if elem in extras else 0)
        assignment.extend([elem] * count)

    assignment = np.array(assignment)
    rng.shuffle(assignment)

    for idx, elem in zip(cations_indices, assignment):
        supercell[idx].symbol = elem

    calculator = ClusterExpansionCalculator(supercell, ce_loaded, scaling=len(supercell) / 2)
    mc_cycles = 200
    interval = n_sites
    results = {}

    for temp in temperatures:
        rng.shuffle(assignment)
        for idx, elem in zip(cations_indices, assignment):
            supercell[idx].symbol = elem

        mc = CanonicalEnsemble(supercell, calculator=calculator, temperature=temp,
                           ensemble_data_write_interval=interval)
        mc.attach_observer(MulticomponentSROObserver(interval=interval, cutoff=3.1, elements=cations))
        mc.run(n_sites * mc_cycles)

        dc = mc.data_container
        df = dc.data
        eq = df.iloc[len(df)//2:]
        results[temp] = {col: eq[col].mean() for col in eq.columns if col.startswith('SRO_')}

        if save:
            import os
            save_path = os.path.join(save, f"mc_mace_T{temp}.dc")
            dc.write(save_path)
            print(f"DataContainer saved to {save_path}")

    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare cluster expansions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ce_src", type=str, required=True,
    )
    parser.add_argument(
        "--save", type=str, default=None,
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mc_ensembler(
        path=args.ce_src,
        save=args.save
    )


if __name__ == "__main__":
    main()
