import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# =================================================================
# Testing data, simplified pi0 -> gamma gamma MC generator
# =================================================================
def MC_generation():
    """
    Create synthetic data for testing
    """

    np.random.seed(42)
    n_events = 1000
    print(f"Total number of events: {n_events}")

    #global synthetic_data
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
            pi0_p = pi0_pt * np.cosh(pi0_eta) # momentum magnitude
            pi0_E = np.sqrt(m_pi0**2 + pi0_p**2)

            # Photon energies (simplified - equal sharing)
            e1_lab = pi0_E / 2
            e2_lab = pi0_E / 2

            # Approximate directions (small opening angle in lab)
            # Opening angle that guarantees m_gg = m_pi0
            dr = 2 * m_pi0 / pi0_E # approximation: theta = m/p

            # Photon directions
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
            pz1 = e1_lab * np.sinh(eta1) / np.cosh(eta1)

            px2 = e2_lab * np.cos(phi2) / np.cosh(eta2)
            py2 = e2_lab * np.sin(phi2) / np.cosh(eta2)
            pz2 = e2_lab * np.sinh(eta2) / np.cosh(eta2)

            px3 = e3 * np.cos(phi3) / np.cosh(eta3)
            py3 = e3 * np.sin(phi3) / np.cosh(eta3)
            pz3 = e3 * np.sinh(eta3) / np.cosh(eta3)

            # Check m_gg
            ee = e1_lab + e2_lab
            pxx = px1 + px2
            pyy = py1 + py2
            pzz = pz1 + pz2
            mass_squared = ee**2 - (pxx**2 + pyy**2 + pzz**2)
            mass_pi0 = np.sqrt(np.maximum(0, mass_squared))
            #print(f"ee  {ee}; mass_pi0  {mass_pi0}")  

            # Data
            synthetic_data.append({
                'event': evt,
                'E1': e1_lab, 'px1': px1, 'py1': py1, 'pz1': pz1,
                'E2': e2_lab, 'px2': px2, 'py2': py2, 'pz2': pz2,
                'E3': e3, 'px3': px3, 'py3': py3, 'pz3': pz3,
                'is_signal': 1,
                'true_pi0_pair': (0, 1) # pion photon index
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
                                e * np.sinh(eta) / np.cosh(eta)   
                ])
        
            synthetic_data.append({
                'event': evt,
                'E1': photons[0], 'px1': photons[1], 'py1': photons[2], 'pz1': photons[3],
                'E2': photons[4], 'px2': photons[5], 'py2': photons[6], 'pz2': photons[7],
                'E3': photons[8], 'px3': photons[9], 'py3': photons[10], 'pz3': photons[11],
                'is_signal': 0,
                'true_pi0_pair': (-1, -1)
            })

    return synthetic_data


# =================================================================
# For a NEW 3-photon event, pick the best pi0 candidate
# =================================================================
def find_best_pi0_candidate(photon_4vectors, model):
    """
    photon_4vectors: list of 3 arrays, each [E, px, py, pz] or [E, pt, eta, phi]
    Returns: (best_pair_indices, probability, mass)
    """
    pairs = [(0,1), (0,2), (1,2)]
    candidates = []

    for i, j in pairs:
        # Calculate EXACT quantities from 4-vectors
        mass = inv_mass_4vector(photon_4vectors[i], photon_4vectors[j])
        #print(f"mass    {mass}; photon1_4vectors    {photon_4vectors[i]}; photon2_4vectors  {photon_4vectors[j]}")

        # True opening angle
        p1 = photon_4vectors[i]
        p2 = photon_4vectors[j]
        p1_mag = np.sqrt(np.maximum(0., p1[1]**2 + p1[2]**2 + p1[3]**2))
        p2_mag = np.sqrt(np.maximum(0., p2[1]**2 + p2[2]**2 + p2[3]**2))
        dot_product = p1[1] * p2[1] + p1[2] * p2[2] + p1[3] * p2[3]
        cos_theta = dot_product / (p1_mag * p2_mag + 1e-10) # 1e-10 avoid divide zero
        theta = np.arccos(np.clip(cos_theta, -1, 1))
        #print(f"theta:  {theta}")

        # Energy asymmetry
        e1, e2 = p1[0], p2[0]
        asym = np.abs(e1 - e2) / (e1 + e2 + 1e-10)
        # print(f"e1: {e1}; e2: {e2}; asym: {asym}")

        # Predict
        proba = model.predict_proba([[mass, theta, cos_theta, asym]])[0, 1]

        candidates.append({
            'pair': (i, j),
            'score': proba,
            'mass': mass,
            'theta': theta
        })

        # Return the best candidate
        best = max(candidates, key=lambda x: x['score']) # best: The entire dictionary with the highest score
        # candidates: A list of dictionaries, each with a 'score' key
        # max(): Built-in Python function that finds the maximum value
        # key=lambda x: x['score']: Tells max() to use the 'score' value for comparison
        return best['pair'], best['score'], best['mass']
    
# =================================================================
# Train XGBoost on EXACT 4-vector quantities
# =================================================================
def train_pi0_classifier_4vector(pair_df):
    """
    Train classifier using EXACT invariant mass and true opening angle.
    This is MORE ACCURATE and SIMPLER than using Delta R approximations.
    """

    # Features: EXACT physics quantities from 4-vectors
    features = ['m_gg', 'opening_angle', 'cos_theta', 'pt_asym']
    X = pair_df[features]
    y = pair_df['is_pi0']

    #print(y[y == 1], "y: ", type(y), "n_ones = ", sum(y == 1)) # print only signal events
    #print(X, "X: ", type(X))
    #print(f"pair_df:    {pair_df.iloc[:, 2]}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2, random_state = 42
    )

    # Ultra-simple XGBoost - shallow trees, fast training
    model = xgb.XGBClassifier(
        n_estimators = 100,
        max_depth = 3, # Shallow = fast, interpretable
        learning_rate = 0.1,
        objective = 'binary:logistic',
        eval_metric = 'auc', # parameter is used in machine learning models (like XGBoost, LightGBM, or sklearn-style APIs) to specify that you want to evaluate your model using AUC (Area Under the ROC Curve) metric.
        use_label_encode=False,
        random_state = 42
    )

    # Train
    model.fit(X_train, y_train,
              eval_set = [(X_test, y_test)],
              verbose=False
    )

    # Evaluate
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    print(f"AUC: {auc:.3f}")

    # Feature importance -  check hat m_gg and opening_angle are top
    importance = model.feature_importances_
    for f, imp in zip(features, importance):
        print(f"    {f}: {imp:.03f}")

    return model

# =================================================================
# Physics METHOD
# =================================================================
def inv_mass_4vector(p1, p2):
    """
    Calculate diphoton invariant mass from two photon 4-vectors.

    Args:
        p1, p2: Arrays/lists of [E, px, py, pz] or [E, pt, eta, phi]

    Returns:
        Invariant mass in GeV
    """

    if len(p1) == 4 and len(p2) == 4:
        e = p1[0] + p2[0]
        px = p1[1] + p2[1]
        py = p1[2] + p2[2]
        pz = p1[3] + p2[3]
        mass_squared = e**2 - (px**2 + py**2 + pz**2)
        # Ensure non-negative before sqrt
        return np.sqrt(np.maximum(0., mass_squared))
        #if (mass_squared < 0):
            #print(f"mass_squared    {mass_squared}")
        #return np.sqrt(mass_squared)
    else:   
        # Use your experiment's Lorentz vector class
        # (e.g., ROOT.TLorentzVector, vector.obj, etc.)
        return (p1 + p2).M()
    
def prepare_3photon_paris(df_events):
    """
    Convert 3-photon events into training paris with EXACT invariant masses.

    Assumes your DataFrame has columns:
    E1, px1, py1, pz1,  # OR E1, pt1, eta1, phi1
    E2, px2, py2, pz2,
    E3, px3, py3, pz3,
    is_signal, true_pi0_pair
    """

    pairs = []

    for _, evt in df_events.iterrows():
        # Get 4-vector for all 3 photons
        # ADAPT THIS TO YOUR EXACT COLUMN NAMES
        photons = [
            np.array([evt.E1, evt.px1, evt.py1, evt.pz1]), # [E, px, py, pz]
            np.array([evt.E2, evt.px2, evt.py2, evt.pz2]),
            np.array([evt.E3, evt.px3, evt.py3, evt.pz3])
        ]

        # All 3 possible pairs
        pair_indices = [(0,1), (0,2), (1,2)]

        #print(f"{pair_indices},{type(pair_indices)}, {type(df_events)}")
        #print(photon)

        for i, j in pair_indices:
            # Calculate EXACT invariant mass from 4-vectors
            mass = inv_mass_4vector(photons[i], photons[j])
            #print(f"mass = {mass}")

            # Opening angle
            p1_mag = np.sqrt(np.maximum(0., photons[i][1]**2 + photons[i][2]**2 + photons[i][3]**2))
            p2_mag = np.sqrt(np.maximum(0., photons[j][1]**2 + photons[j][2]**2 + photons[j][3]**2))
            dot_product = photons[i][1] * photons[j][1] + photons[i][2] * photons[j][2] + photons[i][3] * photons[j][3]
            cos_theta = dot_product / (p1_mag * p2_mag + 1e-10) # 1e-10 avoid divide zero
            theta = np.arccos(np.clip(cos_theta, -1, 1))
            #print(f"p1_mag = {p1_mag}, p2_mag = {p2_mag}")
            #print(f"dot_product = {dot_product}, cos_theta = {cos_theta}")
            #print(f"{np.clip(cos_theta, -1, -1)}")
            #print(f"theta = {theta}")

            # Energy asymmetry
            e1 = photons[i][0]
            e2 = photons[j][0]
            pt_asym = np.abs(e1 - e2) / (e1 + e2 + 1e-10)
            #print(f"p_asym = {pt_asym}")

            # Is this the correct pi0 pair? (require truth info)
            is_pi0 = 0
            if 'is_signal' in evt and evt.is_signal == 1:
                if 'true_pi0_pair' in evt:
                    is_pi0 = 1 if (i, j) == evt.true_pi0_pair else 0
                else:
                    # If you don't have exact pair truth,
                    # assume the pair with mass closest to 0.135 GeV is correct
                    is_pi0 = 1 if abs(mass - 0.135) < 0.015 else 0

            # Data insertion
            pairs.append({
                'event': evt.event,
                'pair_id': f"{evt.event}_{i}{j}",
                'm_gg': mass,
                'opening_angle': theta, # opening angle in radians
                'cos_theta': cos_theta,
                'pt_asym': pt_asym,
                'E1': e1,
                'E2': e2,
                'is_pi0': is_pi0
            })
    
    return pd.DataFrame(pairs)

# =================================================================
# PLOT METHOD
# =================================================================

def plot_hist(df, columns_df, rows, bins, plot_nm):
    
    col_len = len(columns_df) # length of columns of df

    # check col_len
    if (col_len < 0):
        # negative
        print(f"Negative col_len ({col_len})")
        return
    elif (col_len == 0):
        # zero col_len
        print(f"Zero col_len ({col_len})")
    else:
        # postive
        if (col_len % 2 == 0):
            # even case
            print(f"good even col_len ({col_len})")
        else:
            # odd or not integer
            print(f"odd col_len or none integer col_len ({col_len})")
        
    # Create 2x4 subplot grid
    plot_col = int(col_len / rows) # number of rows and columns to the plot
    fig, axes = plt.subplots(rows, plot_col, figsize=(16, 10)) # rows and columns to subplots
    fig.suptitle(r'$\pi^{0}$ photon features', fontsize=16, y=1.02)
    colors = plt.cm.tab10(np.linspace(0, 1, col_len))  # Generate distinct colors
     #color = ["blue", "black", "red", "yellow", "purple", "green", "orange", "brown", "gray", "cyan"]
    #print(f"columns list ({col_len}); plot {rows}x{plot_col}")

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, label in enumerate(columns_df[:col_len]):
        #print(i, label)
        # desity=True normalized
        positive_data = df[df[label] > 0.2][label]
        axes[i].hist(positive_data, color=colors[i], bins=bins, label=label, density=False, edgecolor='black', alpha=0.7)
        #axes[i].hist(df[label], color=colors[i], bins=bins, label=label, density=False, edgecolor='black', alpha=0.7)
        #axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel(None)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best', fontsize=8) 
        #axes[i].set_yscale('log')  # <-- LOG SCALE ON Y-AXIS
        
    plt.tight_layout()
    plt.savefig('./plots/' + plot_nm + '.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.close()

