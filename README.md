#  High-Entropy Ceramics — MLIP Benchmarking Pipeline

> **Generate → Relax → Compare**: A computational pipeline for studying high-entropy ceramic (HEC) rock-salt structures using Machine-Learned Interatomic Potentials (MLIPs).

---

## Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Pipeline Details](#-pipeline-details)
- [Key Concepts Explained](#-key-concepts-explained)
- [Learning Resources](#-learning-resources)
- [License](#-license)


## Installation

> [!IMPORTANT]
> **fairchem-core** models require a **separate checkpoint download**. After installing `fairchem-core`, follow the [fairchem documentation](https://fair-chem.github.io/) to download the EquiformerV2 checkpoint (e.g., `EquiformerV2-31M-S2EF-OC20-All+MD`). The checkpoint path must be provided when initializing the calculator.

### Install dependencies
```bash
pip install -r requirements.txt
```
## Quick Start

```bash
# Step 1: Generate 1000 random HEC structures
python structure-sim.py --n-structures 1000 --seed 42

# Step 2: Relax with CHGNet
python relax_structures.py --calculator chgnet \
    --input structures_unrelaxed.traj \
    --output structures_chgnet.traj

# Step 3: Relax with MACE-MP-0
python relax_structures.py --calculator mace \
    --input structures_unrelaxed.traj \
    --output structures_mace.traj

# Step 4: Compare the two MLIPs
python compare_mlips.py \
    --files structures_chgnet.traj structures_mace.traj \
    --labels CHGNet MACE
```

---

## Pipeline Details

### 1️⃣ `structure-sim.py` — Structure Generation

Generates random high-entropy ceramic supercells by decorating MgO rock-salt cation sites.

**What it does:**

1. Builds a **MgO rock-salt** primitive cell (a = 4.21 Å) using ASE.
2. Creates a **4×4×4 supercell** (64 cation sites + 64 oxygen sites = 128 atoms).
3. For each of the 1000 structures, **randomly assigns** one of {Mg, Co, Ni, Cu, Zn} to each cation site.
4. Applies a small **random displacement** (`rattle`, σ = 0.05 Å) to break symmetry — this helps optimizers find true local minima rather than saddle points.
5. Saves all structures to `structures_unrelaxed.traj` (ASE trajectory format).

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-structures` | 1000 | Number of random configurations to generate |
| `--seed` | 42 | Random seed for reproducibility |
| `--elements` | `Mg Co Ni Cu Zn` | Cation element pool |
| `--supercell` | `4 4 4` | Supercell dimensions |
| `--mode` | `equimolar` | Composition mode: `equimolar` or `random` |
| `--rattle-stdev` | 0.05 | Rattle displacement σ (Å) |
| `--lattice-constant` | 4.21 | Rock-salt lattice constant (Å) |
| `--output` | `structures_unrelaxed.traj` | Output trajectory file |

---

### 2️⃣ `relax_structures.py` — Structural Relaxation

Relaxes all generated structures using a chosen MLIP calculator.

**What it does:**

1. Reads unrelaxed structures from the input `.traj` file.
2. Initializes the chosen MLIP calculator (CHGNet, MACE, or EquiformerV2).
3. For each structure:
   - Wraps the `Atoms` object in an **`ExpCellFilter`** (or `FrechetCellFilter`) so that both atomic positions *and* the unit cell shape/volume are optimized.
   - Runs the **BFGS** (or L-BFGS or FIRE) optimizer until the maximum force falls below `fmax`.
4. Saves all relaxed structures to the output `.traj` file.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--calculator` | *(required)* | MLIP to use: `chgnet`, `mace`, or `eqv2` |
| `--input` | `structures_unrelaxed.traj` | Input trajectory file |
| `--output` | `structures_<calculator>.traj` | Output trajectory file |
| `--fmax` | 0.05 | Max force convergence threshold (eV/Å) |
| `--max-steps` | 200 | Maximum optimizer steps per structure |
| `--optimizer` | `LBFGS` | Optimizer: `LBFGS` or `FIRE` |
| `--no-cell-relax` | *(flag)* | Disable cell relaxation (positions only) |
| `--device` | `auto` | Compute device: `cpu`, `cuda`, or `auto` |
| `--max-structures` | *(all)* | Only relax first N structures (for testing) |

**Example:**

```bash
# Relax with MACE on GPU, stricter convergence
python relax_structures.py \
    --calculator mace \
    --input structures_unrelaxed.traj \
    --output structures_mace_tight.traj \
    --fmax 0.01 \
    --device cuda
```

---

### 3️⃣ `compare_mlips.py` — MLIP Comparison & Analysis

Compares relaxed structures from different MLIPs side by side.

**What it does:**

1. Loads two or more `.traj` files (one per MLIP).
2. For each structure pair, computes:
   - **Energy per atom** (eV/atom) — total potential energy normalized by atom count.
   - **Volume per atom** (ų/atom) — unit cell volume normalized by atom count.
   - **RMSD** — Root Mean Square Displacement between relaxed positions (measures how differently the MLIPs relax the same starting structure).
   - **Force distributions** — histograms of residual forces at convergence.
3. Generates **publication-quality plots** (matplotlib) showing distributions, correlations, and parity plots.
4. Prints summary statistics (mean, std, MAE between MLIPs).

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--files` | *(required)* | Space-separated list of `.traj` files to compare |
| `--labels` | *(required)* | Human-readable labels for each file (same order) |
| `--outdir` | `comparison_plots/` | Directory for saving figures |
| `--reference` | *(none)* | Optional unrelaxed reference trajectory |

**Example:**

```bash
# Compare three MLIPs
python compare_mlips.py \
    --files structures_chgnet.traj structures_mace.traj structures_eqv2.traj \
    --labels CHGNet MACE EqV2 \
    --outdir results/
```

## Key Concepts Explained

### What are ExpCellFilter / FrechetCellFilter?

By default, ASE optimizers only move **atomic positions** within a fixed unit cell. To also optimize the **cell shape and volume** (crucial for HECs, since different cation mixes may prefer different lattice parameters), we wrap the `Atoms` object in a **cell filter**:

```python
from ase.filters import FrechetCellFilter    # or ExpCellFilter
from ase.optimize import BFGS

# This allows the optimizer to change both positions AND cell
filtered_atoms = FrechetCellFilter(atoms)
optimizer = BFGS(filtered_atoms)
optimizer.run(fmax=0.05)
```

- **`ExpCellFilter`** — Maps cell degrees of freedom via the matrix exponential. Widely used and well-tested.
- **`FrechetCellFilter`** — A newer alternative that uses the Fréchet derivative. Can be more numerically stable for severely distorted cells.

Both expose the 6 cell degrees of freedom (3 lengths + 3 angles) as "virtual forces" that the optimizer can act on, alongside the 3N atomic position forces.

### What is `fmax`?

**`fmax`** is the convergence criterion for structural relaxation. It is the **maximum component of the force** (in eV/Å) on any atom in the structure. The optimizer stops when:

```
max(|Fᵢ|) < fmax    for all atoms i
```

Typical values:

| fmax (eV/Å) | Meaning |
|-------------|---------|
| 0.05 | Standard convergence — good for screening |
| 0.01 | Tight convergence — for publication-quality results |
| 0.001 | Very tight — for phonon calculations, elastic constants |

Lower `fmax` → more accurate structure, but more optimizer steps → slower.

---
## Learning Resources

### ASE (Atomic Simulation Environment)

ASE is the backbone of this project. It provides Python objects for atoms, calculators, optimizers, and I/O.

| Resource | 
|----------|
| [Documentation](https://wiki.fysik.dtu.dk/ase/) |
| [Calculators Tutorial](https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html) | 
| [Optimization Tutorial](https://wiki.fysik.dtu.dk/ase/ase/optimize.html) | 
| [Constraints & Filters](https://wiki.fysik.dtu.dk/ase/ase/constraints.html) | 
| [I/O](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) | 

---

### CHGNet

**CHGNet** (Crystal Hamiltonian Graph Neural Network) is a pretrained universal MLIP from the Ceder Group at UC Berkeley. It uniquely incorporates **magnetic moments** and **charge states**, making it particularly well-suited for transition-metal oxides like HECs.

| Resource | 
|----------|
| [Paper](https://doi.org/10.1038/s42256-023-00716-3)|
| [GitHub Repository](https://github.com/CederGroupHub/chgnet) |
| [Examples](https://github.com/CederGroupHub/chgnet/tree/main/examples) | 

**Key concepts:** Pretrained on Materials Project relaxation trajectories; handles charge-decorated graphs; includes stress tensors for cell optimization; ~400K parameters in the default model.

---

### MACE

**MACE** (Multi-ACE) uses higher-order equivariant message passing to achieve high accuracy with fewer message-passing layers. **MACE-MP-0** is their foundation model, pretrained on the Materials Project.

| Resource |
|----------|
| [Intuittion Paper](https://arxiv.org/abs/2206.07697) |
| [Model Paper](https://arxiv.org/abs/2401.00096) | 
| [GitHub Repository](https://github.com/ACEsuit/mace) |
| [Matbench Discovery](https://matbench-discovery.materialsproject.org/) | 

**Key concepts:** Equivariant message passing using body-ordered (multi-body) interactions; E(3)-equivariant by construction; MACE-MP-0 trained on ~150K Materials Project structures; available in small/medium/large variants.

---

### fairchem / Open Catalyst Project

**fairchem** (formerly Open Catalyst Project) from Meta FAIR provides state-of-the-art GNN models trained on massive catalysis datasets (OC20: 260M DFT calculations). Models like **EquiformerV2** and **EScAIP** achieve top accuracy across multiple benchmarks.

| Resource | 
|----------|
| [GitHub Repository](https://github.com/FAIR-Chem/fairchem) |
| [Documentation](https://fair-chem.github.io/) |
| [EquiformerV2 Paper](https://arxiv.org/abs/2306.12059) | 
| [Model Zoo](https://fair-chem.github.io/core/model_checkpoints.html) |

**Key concepts:** Trained on OC20/OC22 datasets (primarily catalysis, but transferable to bulk materials); EquiformerV2 uses equivariant Transformers with SO(3) representations; requires separate checkpoint download after `pip install`.

> [!NOTE]
> fairchem models were originally designed for catalysis (surface reactions), but their training data is diverse enough to handle bulk materials like HECs. Performance may vary — that's exactly what this project benchmarks!


## 📄 License

*Add your license here (e.g., MIT, Apache 2.0, etc.)*

---

