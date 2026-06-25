# Usage    
# python src/analyze_mc.py --dc_dir results/mc_ensembler/
import argparse
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from mchammer import DataContainer


def load_dc(filepath):
    dc = DataContainer.read(filepath)
    return dc.data


def get_equilibrated(df, cutoff=0.8):
    return df.iloc[int(len(df) * cutoff):]


def plot_energy_and_sro_evolution(df, elements, temperature=None, plot_dir="results/mc_ensembler/plots"):
    # ---------------------------------------------------------
    # PLOT 1 & 2: Trajectories over Time
    # ---------------------------------------------------------
    title_suffix = f" (T={temperature} K)" if temperature else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Potential Energy Curve
    axes[0].plot(df['mctrial'], df['potential'], color='black', linewidth=1.5)
    axes[0].set_xlabel('Monte Carlo Trials')
    axes[0].set_ylabel('Potential Energy (eV)')
    axes[0].set_title(f'Energy Equilibration{title_suffix}')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # SRO Evolution (self-pairs only to avoid 25 overlapping lines)
    sample_sros = [f"SRO_{e}_{e}" for e in elements]
    for sro in sample_sros:
        if sro in df.columns:
            smoothed = df[sro].rolling(window=max(1, len(df) // 50)).mean()
            axes[1].plot(df['mctrial'], smoothed, label=sro)

    axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Monte Carlo Trials')
    axes[1].set_ylabel('Warren-Cowley SRO (\u03B1)')
    axes[1].set_title(f'SRO Evolution (Self-Interactions){title_suffix}')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"energy_sro_evolution_T{temperature}.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_sro_heatmap(df, elements, temperature=None, plot_dir="results/mc_ensembler/plots"):
    # ---------------------------------------------------------
    # PLOT 3: Equilibrated SRO Matrix Heatmap
    # ---------------------------------------------------------
    title_suffix = f" (T={temperature} K)" if temperature else ""
    eq_df = get_equilibrated(df)

    sro_matrix = pd.DataFrame(index=elements, columns=elements, dtype=float)
    for e_a in elements:
        for e_b in elements:
            col_name = f"SRO_{e_a}_{e_b}"
            if col_name in eq_df.columns:
                sro_matrix.loc[e_a, e_b] = eq_df[col_name].mean()
            else:
                sro_matrix.loc[e_a, e_b] = 0.0

    plt.figure(figsize=(7, 6))
    sns.heatmap(sro_matrix, annot=True, cmap="coolwarm", center=0,
                fmt=".3f", cbar_kws={'label': 'Warren-Cowley SRO'})
    plt.title(f'Averaged SRO Matrix (Final 20% of Simulation){title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"sro_heatmap_T{temperature}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sro_vs_temperature(results, elements, plot_dir="results/mc_ensembler/plots"):
    # ---------------------------------------------------------
    # PLOT 4: SRO vs Temperature
    # ---------------------------------------------------------
    temperatures = sorted(results.keys())
    df = pd.DataFrame({t: results[t] for t in temperatures}).T
    df.index.name = 'Temperature (K)'

    # -- Self-interactions --
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    self_pairs = [f"SRO_{e}_{e}" for e in elements]
    for col in self_pairs:
        if col in df.columns:
            axes[0].plot(temperatures, df[col], marker='o', label=col)

    axes[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('Temperature (K)')
    axes[0].set_ylabel('Warren-Cowley SRO (\u03B1)')
    axes[0].set_title('Self-Interaction SRO vs Temperature')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # -- Cross-interactions (unique pairs only) --
    seen = set()
    cross_pairs = []
    for e_a in elements:
        for e_b in elements:
            if e_a != e_b:
                key = tuple(sorted([e_a, e_b]))
                if key not in seen:
                    seen.add(key)
                    cross_pairs.append(f"SRO_{e_a}_{e_b}")

    for col in cross_pairs:
        if col in df.columns:
            axes[1].plot(temperatures, df[col], marker='o', label=col)

    axes[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('Temperature (K)')
    axes[1].set_ylabel('Warren-Cowley SRO (\u03B1)')
    axes[1].set_title('Cross-Interaction SRO vs Temperature')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "sro_vs_temperature.png"), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_single(filepath, elements):
    df = load_dc(filepath)
    print(df.columns.tolist())

    # infer temperature from filename e.g. mc_mace_T1000.dc
    temperature = None
    basename = os.path.basename(filepath)
    if '_T' in basename:
        try:
            temperature = int(basename.split('_T')[-1].replace('.dc', ''))
        except ValueError:
            pass

    plot_energy_and_sro_evolution(df, elements, temperature)
    plot_sro_heatmap(df, elements, temperature)


def analyze_multi_temperature(dc_dir, elements, plot_dir="results/mc_ensembler/plots"):
    os.makedirs(plot_dir, exist_ok=True)
    pattern = os.path.join(dc_dir, "mc_mace_T*.dc")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files matching 'mc_mace_T*.dc' found in {dc_dir}")
        return

    results = {}
    for filepath in files:
        basename = os.path.basename(filepath)
        try:
            temperature = int(basename.split('_T')[-1].replace('.dc', ''))
        except ValueError:
            print(f"Skipping {basename}: could not parse temperature")
            continue

        print(f"Loading T={temperature} K: {filepath}")
        df = load_dc(filepath)
        eq_df = get_equilibrated(df)

        plot_energy_and_sro_evolution(df, elements, temperature)
        plot_sro_heatmap(df, elements, temperature)

        results[temperature] = {
            col: eq_df[col].mean()
            for col in eq_df.columns if col.startswith('SRO_')
        }

    if len(results) > 1:
        plot_sro_vs_temperature(results, elements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize mchammer DataContainer results.")
    parser.add_argument("--dc", type=str, default=None, help="Path to a single .dc file")
    parser.add_argument("--dc_dir", type=str, default=None, help="Directory containing mc_mace_T*.dc files")
    parser.add_argument("--plot_dir", type=str, default="results/mc_ensembler/plots")
    args = parser.parse_args()

    # Define your HEC cation pool
    cations = ["Co", "Cu", "Mg", "Ni", "Zn"]

    if args.dc:
        analyze_single(args.dc, cations)
    elif args.dc_dir:
        analyze_multi_temperature(args.dc_dir, cations, plot_dir=args.plot_dir)
    else:
        print("Provide either --dc or --dc_dir")
