import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
from methods import prepare_3photon_paris, plot_hist, find_best_pi0_candidate, train_pi0_classifier_4vector, MC_generation
import seaborn as sns

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

    synthetic_data = MC_generation()
    synthetic_df = pd.DataFrame(synthetic_data) 
    print(synthetic_df.head(5))

    df = pd.DataFrame(synthetic_data) # read data to DataFrame: 1. synthetic_data (testing data); 2. kloe_data (MC and data from the KLOE 3pi analysis)
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
    
    # plot photon pair features
    plot_hist(pair_df, columns_pair_df, 2, 100, "pair_features")

    print("\n1. Inspect features ...")
    print(f"# columns ({len(pair_df.columns[:])}); Column name:    {pair_df.columns[:]}")
    #print(pair_df.head(5))
    #print(f"Statistics:\n{pair_df.describe()}")
    #print(df.head(10))
    #print(f"Column name: {df.columns[0]}")
    #print(f"Data type: {df.dtypes}")
    #print(f"Statistics:\n{df.describe()}")
    #print("data:", df.shape())

    # Paire plot (Scatter Matrix) correlations
    sample_size = min (1000, len(pair_df))
    sample_df = pair_df.sample(sample_size)
    print(sample_df.head(5))

    # Create pair plot
    #('m_gg', 'opening_angle'),      # Should be correlated (higher mass = larger opening angle)
    #('m_gg', 'pt_asym'),             # Check if mass correlates with asymmetry
    #('opening_angle', 'pt_asym'),    # Opening angle vs energy asymmetry
    #('cos_theta', 'opening_angle'),  # These are directly related
    #('E1', 'E2'),                    # Photon energies (should be anti-correlated for π⁰)
    feature_columns = ['m_gg', 'opening_angle', 'cos_theta', 'pt_asym', 'E1', 'E2']
    g = sns.pairplot(sample_df[feature_columns + ['is_pi0']], # Data
                     hue = 'is_pi0', # Color grouping, points by the values in the 'is_pi0' column
                     palette={1: 'blue', 0: 'red'}, # 3. colors     
                     diag_kind='hist', # Diagonal plot type
                     plot_kws={'alpha': 0.5, 's': 10}, # Scatter plot options
                     diag_kws={'alpha': 0.7, 'edgecolor': 'black'} # Histogram options  
    )
    g.figure.suptitle('Feature Pair Plot (Signal=Blue, Background=Red)', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.savefig('./plots/feature_pairplot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Check signal pi0 mass

    signal_pi0_mass = pair_df[(pair_df['is_pi0'] == 1) & (pair_df['m_gg'] > 0.1)]['m_gg'].tolist()
    #print(signal_pi0_mass)
    plt.hist(signal_pi0_mass, color='Black', bins=100, density=False, edgecolor='black', alpha=0.7)
    plt.xlabel(r'$m_{\gamma\gamma}$ (GeV)')
    plt.ylabel('Events')
    plt.title(f'π⁰ Mass Distribution (n={len(signal_pi0_mass)})')
    plt.grid(True, alpha=0.3)
    plt.savefig('./plots/signal_pi0.png')
    plt.show(block=False)
    plt.close()
        
    # Train classifier
    print("\n2. Training XGBoost on EXACT 4-vector features ...")
    model = train_pi0_classifier_4vector(pair_df)

    # Test on a few events
    print("\n3. Testing on 5 random events:")
    test_events = df.sample(50)
    for _, evt in test_events.iterrows():
        # evt is a list object
        photons = [
            np.array([evt.E1, evt.px1, evt.py1, evt.pz1]),
            np.array([evt.E2, evt.px2, evt.py2, evt.pz2]),
            np.array([evt.E3, evt.px3, evt.py3, evt.pz3]),
        ]

        best_pair, score, mass = find_best_pi0_candidate(photons, model)
        truth = f"True pi0: {evt.true_pi0_pair}" if evt.is_signal else "Background event"
        #print(f"   Event {evt.event}: Best pair {best_pair}, score={score:.3f}, m={mass:.3f} | {truth}")