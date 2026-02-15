import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        e = p1[1] + p2[0]
        px = p1[1] + p2[1]
        py = p1[2] + p2[2]
        pz = p1[3] + p2[3]
        mass_squared = e**2 - (px**2 + py**2 + pz**2)
        # Ensure non-negative before sqrt
        return np.sqrt(np.maximum(0., mass_squared))
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
    #print(f"columns list ({col_len}); plot {rows}x{plot_col}")

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, label in enumerate(columns_df[:col_len]):
        print(i, label)
        axes[i].hist(df[label], color=colors[i], bins=bins, label=label,density=True, edgecolor='black', alpha=0.7)
        #axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel(label)
        axes[i].set_ylabel(None)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='best', fontsize=8) 
    plt.tight_layout()
    plt.savefig(plot_nm + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

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
'''