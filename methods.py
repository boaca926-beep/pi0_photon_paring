import pandas as pd
import numpy as np

def test_print():
    print("\n All methods used in pi0 photon paring are stored here!")

'''
    Calculate invariant mass CORRECTLY from 4-vectors
'''
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
    


'''
    For 3-photon events, create ALL pairs with EXACT masses
'''
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
            theta = np.arccos(np.clip(cos_theta, -1, -1))

            # Energy asymmetry

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
                'cos_theta': cos_theta,
                'is_pi0': is_pi0
            })
    
    return pd.DataFrame(pairs)
