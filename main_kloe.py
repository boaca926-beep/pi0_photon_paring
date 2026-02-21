import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
from methods import prepare_3photon_paris, plot_hist, find_best_pi0_candidate, train_pi0_classifier_4vector, MC_generation
from methods import kloe_sample, plot_compr_hist
import seaborn as sns

if __name__ == "__main__":

    all_signal, good_signal, bad_signal = kloe_sample()
    #print(good_E1.head(5))
    print(f'len bad     ', (all_signal['true_pi0_pair'] == (-1, -1)).sum())
    print(all_signal['true_pi0_pair'].value_counts(), f'len good    {len(good_signal)}')

    #nb_all_signal = [i for i in range(len(all_signal))] # Vector of number of all signal events
    #all_signal['event'] = nb_all_signal
    print(f'all_signal\n    ', all_signal.head(6))
    print(f'good_signal\n   ', good_signal.head(6))
    print(f'bad_signal\n   ', bad_signal.head(6))

    #columns = [col for col in all_signal.columns]
    all_signal_plot = all_signal.drop(['event', 'true_pi0_pair'], axis=1) # Ready for plot
    good_signal_plot = good_signal.drop(['event'], axis=1)
    bad_signal_plot = bad_signal.drop(['event'], axis=1)
    #print(all_signal_plot.head(8))
    #print(good_signal.head(8))

    #plot_hist(all_signal_plot, 3, 100, "Photon 4-momentum (all signal)") # data, column list, number of rows to plot, number of bins
    #plot_hist(good_signal_plot, 3, 100, "Photon 4-momentum (good signal)") 
    #plot_hist(bad_signal_plot, 3, 100, "Photon 4-momentum (bad signal)") 

    df_set = [all_signal_plot, good_signal_plot, bad_signal_plot]
    plot_compr_hist(df_set, 3, 100, "Photon 4-momentum") # Comparison plot

    synthetic_data = MC_generation()
    synthetic_df = pd.DataFrame(synthetic_data) 
    #print(synthetic_df.head(5))
    