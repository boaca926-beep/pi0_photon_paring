import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import warnings
from methods import prepare_3photon_paris, plot_hist

# Based on description from xgboost_pi0_photon_paring

# Photon1_eta is the pseudorapidity of the first photon. dR (or dr) is the angular distance between two photons.
# Pseudorapidity: Angle relative to the beam pipe. η=0 = perpendicular to beam; η=±∞ = along the beam.
# η = -ln(tan(θ/2)), where θ is the polar angle from the beam axis. In your ntuple, it's just a number, typically between -4.0 and +4.0 for LHC detectors, or -1.0 to +1.0 for Belle II style detectors.
# Example: photon1_eta = 0.5 means the first photon hit the calorimeter at η=0.5.
# Calorimeters are segmented in η. Photons from π⁰ decay have small Δη

# Angular distance: ΔR or dR, How far apart two photons are in the detector, π⁰ → γγ photons are very close together (ΔR ~ 0.01-0.1). Random photon pairs are far apart (ΔR ~ 0.5-3.0)
# ΔR = √[(Δη)² + (Δφ)²], η = difference in pseudorapidity between the two photons; Δφ = difference in azimuthal angle (wrapped to [-π, π])
# def deltaR(eta1, phi1, eta2, phi2):
#    dphi = np.abs(phi1 - phi2)
#    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)  # Wrap around 2π
#    return np.sqrt((eta1 - eta2)**2 + dphi**2)

# Photon pair type	Typical ΔR	Typical mass	XGBoost learns
# Real π⁰	0.01 - 0.10 (very close)	~0.135 GeV	"Small ΔR + mass ~0.135 = π⁰"
# Random background	0.20 - 3.00 (far apart)	Random	"Large ΔR = fake π⁰"
# This is why your 3-photon problem is actually easy: In any event with 3 photons, the real π⁰ pair will have ΔR ~10x smaller than the two fake pairs involving the extra photon.

# import uproot
# file = uproot.open("your_file.root")
# tree = file["your_tree_name"]
# print(tree.keys())  # THIS SHOWS ALL COLUMN NAMES

# For true π⁰ → γγ events
# dR_values = []
# for event in your_data:
#     if event.is_true_pi0:
#         dR = deltaR(event.photon1_eta, event.photon1_phi,
#                   event.photon2_eta, event.photon2_phi)
#         dR_values.append(dR)

# print(f"π⁰ dR: mean={np.mean(dR_values):.3f}, max={np.max(dR_values):.3f}")
# If you see mean dR < 0.2, you're good. If mean dR > 0.5, you're using the wrong columns.
# These are the two most basic geometric quantities in any collider physics analysis. Here is the exact definition:

# Find the uv Python interpreter
# > uv run python -c "import sys; print(sys.executable)"
# add manually using Ctrl+P

print("pi0 photon selction, reduce combinatorial background.")

if __name__ == "__main__":
    print("="*50)
    print("SIMPLE XGBOOST FOR 3-PHOTON pi0 RECONSTRUCTION")
    print("="*50)

    # Create synthetic data for testing
    np.random.seed(42)
    n_events = 1000
    print(f"Total number of events: {n_events}")
 
    global synthetic_data
    synthetic_data = []
    for evt in range(n_events):
        # Signal: one pi0 + one extra photon
        if np.random.random() < 0.5:
            # pi0 at rest in its own frame, then boosted
            m_pi0 = 0.135
            #print(f"pi0 mass = {m_pi0} GeV")

            # Generate pi0 with some momentum
            pi0_pt = np.random.uniform(1, 10)
            pi0_eta = np.random.uniform(-1, 1)
            pi0_phi = np.random.uniform(-np.pi, np.pi)
            #print(f"pi0: (pt, eta, phi) = ({pi0_pt}, {pi0_eta}, {pi0_phi})")

            # Decay in rest frame: back-to-back photons
            # Then boost to lab frame
            # This is simplifed - in reality use proper decay generator
            # pi0 photon pair
            e1_lab = pi0_pt * np.cosh(pi0_eta) / 2
            e2_lab = pi0_pt * np.cosh(pi0_eta) / 2

            # Approximate directions (small opening angle in lab)
            dr = 0.135 / pi0_pt # approximation: theta = m/p
            phi1 = pi0_phi
            phi2 = pi0_phi + dr
            eta1 = pi0_eta
            eta2 = pi0_eta + dr / 2

            # Extra random photon
            e3 = np.random.uniform(0.5, 5)
            eta3 = np.random.uniform(-2, 2)
            phi3 = np.random.uniform(-np.pi, np.pi)

            # Convert (pt, eta, phi) to (E, px, py, pz)
            px1 = e1_lab * np.cos(phi1) / np.cosh(eta1)
            py1 = e1_lab * np.sin(phi1) / np.cosh(eta1)
            pz1 = e1_lab * np.sinh(eta1)

            px2 = e2_lab * np.cos(phi2) / np.cosh(eta2)
            py2 = e2_lab * np.sin(phi2) / np.cosh(eta2)
            pz2 = e2_lab * np.sinh(eta2)

            px3 = e3 * np.cos(phi3) / np.cosh(eta3)
            py3 = e3 * np.sin(phi3) / np.cosh(eta3)
            pz3 = e3 * np.sinh(eta3)

            # Data
            synthetic_data.append({
                'event': evt,
                'E1': e1_lab, 'px1': px1, 'py1': py1, 'pz1': pz1,
                'E2': e2_lab, 'px2': px2, 'py2': py2, 'pz2': pz2,
                'E3': e3, 'px3': px3, 'py3': py3, 'pz3': pz3,
                'is_signal': 1,
                'true_pi0_pair': (0,1)
            })
            # print(f"e1 = {e1}, e1 = {e2}, dr = {dr}, e3 = {e3}, dr_extra = {dr_extra}")
        else:
            # Background: three random photon final state
            photons = []
            for k in range(3):
                e = np.random.uniform(0.5, 10)
                eta = np.random.uniform(-2, 2)
                phi = np.random.uniform(-np.pi, np.pi)
                photons.extend([e,
                                e * np.cos(phi) / np.cosh(eta),
                                e * np.sin(phi) / np.cosh(eta),
                                e * np.sinh(eta)    
                ])
        
            synthetic_data.append({
                'event': evt,
                'E1': photons[0], 'px1': photons[1], 'py1': photons[2], 'pz1': photons[3],
                'E2': photons[4], 'px2': photons[5], 'py2': photons[6], 'pz2': photons[7],
                'E3': photons[8], 'px3': photons[9], 'py3': photons[10], 'pz3': photons[11],
                'is_signal': 0,
                'true_pi0_pair': (-1, -1)
            })

    df = pd.DataFrame(synthetic_data)
    # columns list with labels only related to photon 4-momentum
    columns_df = [col for col in df.columns if col not in ['event', 'is_signal', 'true_pi0_pair']]
    print(f"Generated {len(df)} events ({df.is_signal.sum()}) signal, {df.is_signal.sum()} background")
    print(f"photon 4-mom data set {df.head(5)}")
    print(f"photon features: {columns_df}")
    #print(f"photon 4-mom (E, px, py, pz): ({df[]})")

    # plot 3 photon 4-momentum
    plot_hist(df, columns_df, 3, 100, "photon_features") # data, column list, number of rows to plot, number of bins

     # Prepare pair dataset with EXACT 4-vector quantities
    print("\n0. Creating photon pairs with EXACT invariant masses ...")
    pair_df = prepare_3photon_paris(df)
    columns_pair_df = [col for col in pair_df.columns if col != 'pair_id']

    #print(f"    {len(pair_df)} pairs created")
    #print(f"    {pair_df.is_pi0.sum()} true pi0 pairs")
    print(f"    Cloumns: {columns_pair_df}")
    plot_hist(pair_df, columns_pair_df, 2, 100, "pair_features")

    # Feature ispection: EXACT physics quantities from 4-vectors
    features = ['m_gg']
    X = pair_df[features]

    #print(X, "X: ", type(X))
    #print(f"pair_df:    {pair_df.iloc[:, 2]}")

    print("\n1. Inspect features ...")
    #print(f"# columns ({len(pair_df.columns[:])}); Column name:    {pair_df.columns[:]}")
    #print(pair_df.head(5))
    print(f"Statistics:\n{pair_df.describe()}")
    #print(df.head(10))
    #print(f"Column name: {df.columns[0]}")
    #print(f"Data type: {df.dtypes}")
    #print(f"Statistics:\n{df.describe()}")
    #print("data:", df.shape())

    ## Plot pi0 photon features
    
    r'''
    ## Plot pi0 features
    columns_to_plot = [col for col in pair_df.columns if col != 'pair_id']
    n_cols = len(columns_to_plot)
    print(f"columns_to_plot ({n_cols}): {columns_to_plot}")
    #color = ["blue", "black", "red", "yellow", "purple", "green", "orange", "brown", "gray", "cyan"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns_to_plot)))  # Generate distinct colors
    
    # Create 2x4 subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 10)) # 2 rows, 4 columns
    fig.suptitle(r'$\pi^{0}$ photon features', fontsize=16, y=1.02)

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    bins = 100
    for i, label in enumerate(columns_to_plot[:8]):
        print(i, label)
        axes[i].hist(pair_df[label], color=colors[i], bins=bins, density=True, edgecolor='black', alpha=0.7)
        axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel(None)
        axes[i].grid(True, alpha=0.3)

    plt.savefig('features.png', dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()

    # Train classifier
    print("\n2. Training XGBoost on EXACT 4-vector features ...")
    '''