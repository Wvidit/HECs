# Usage
# python src/training_ce.py --initialfile 

from ase.io import read
import argparse

def explore(path):
    from ase.db import connect
    from icet import ClusterSpace
    from ase.build import bulk
    
    db = connect(path)
    primitive = bulk("MgO", crystalstructure="rocksalt", a=4.21)
    cs = ClusterSpace(structure=primitive,
                      cutoffs=[6.0, 3.0],
                      chemical_symbols=['Mg', 'Co', 'Ni', 'Cu', 'Zn'])
    print(primitive)
    print(cs)

def load_training(path):
    from icet import StructureContainer
    from ase.db import connect
    from icet import ClusterSpace
    from ase.build import bulk
    import numpy as np
    from icet import ClusterExpansion
    from trainstation import CrossValidationEstimator
    
    db = connect(path, type='db')
    primitive = bulk("MgO", crystalstructure="rocksalt", a=4.21)
    cs = ClusterSpace(structure=primitive,
                      cutoffs=[6.0, 3.0],
                      chemical_symbols=[['Mg', 'Co', 'Ni', 'Cu', 'Zn'],['O']])
    sc = StructureContainer(cluster_space=cs)
    for row in db.select():
        data = row.data
        energy = data['properties']['energy']
        sc.add_structure(structure=row.toatoms(), properties={'energy': energy})

    A, y = sc.get_fit_data(key='energy')
    M = np.array(cs.get_multiplicities())

    A = A*(M.T)
    fit_kwargs = dict(threshold_lambda = 1e5)
    opt = CrossValidationEstimator(fit_data=(A,y), fit_method='ardr', **fit_kwargs)

    opt.train()
    opt.validate() 
    print(opt)
    parameters = opt.parameters.copy()

    parameters *= M
    ce_regular = ClusterExpansion(cluster_space=cs, parameters=parameters)
    ce_regular.write(path.split('/')[-1].split('.')[0] + '.tar')
# Load using
# from icet import ClusterExpansion
# ce_loaded = ClusterExpansion.read("my_ce.tar")

def parse_args(argv=None):
    parser=argparse.ArgumentParser()

    parser.add_argument(
            "--initialfile", type=str,
    )
    parser.add_argument(
            "--finalfile", type=str,
    )
    parser.add_argument(
            "--samplesize",
    )

    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    #explore(args.initialfile)
    load_training(args.initialfile)


if __name__ == '__main__':
    main()
