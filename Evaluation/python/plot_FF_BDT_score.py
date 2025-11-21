import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import numpy as np
import yaml
from statsmodels.stats.weightstats import DescrStatsW
from plot_utils import stacked_histogram
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Classifier Evaluation")
    parser.add_argument('--channel', type=str, help="Channel to train", required=True)
    return parser.parse_args()

# Plotting style
plt.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 14})

# Luminosity
lumi = 61.90

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams


def plot_score(cfg, parity, channel):
    # Load the model predictions
    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)
    pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))
    process_id_counts = pred_df['process_id'].value_counts()

    # Split into individual classes (by process ID)
    # genuine taus
    DY_tau = pred_df.loc[pred_df['process_id'] == 11]
    DY_lep = pred_df.loc[pred_df['process_id'] == 12]
    Top = pred_df.loc[(pred_df['process_id'] == 21)]
    VV = pred_df.loc[(pred_df['process_id'] == 51) | (pred_df['process_id'] == 31)]
    # jet fakes from W
    W = pred_df.loc[(pred_df['process_id'] == 43)]

    bins = np.arange(0, 1.025, 0.025)


    df_tau = pd.concat([DY_tau, DY_lep, Top, VV])
    w_total = np.sum(W['weight'])
    for cut in np.arange(0, 1.05, 0.05):
        w_count = np.sum(W['weight'][(W['pred_1'] > cut)])
        t_count = np.sum(df_tau['weight'][(df_tau['pred_1'] > cut)])
        total_count = w_count + t_count

        print(f"At CUT: {cut} - W efficiency: {round(w_count/w_total*100,3)}, W purity: {round(w_count/total_count*100,3)}")


    # Plot the optimised distribution
    print(f"Plotting score distribution ")
    fig, ax = plt.subplots(figsize = (6,6))
    # Stacked histogram
    histo = stacked_histogram("pred_1", ax, bins)

    # Add fake processes
    histo.add_bkg(W, "WJets")
    # genuine backgrounds
    histo.add_bkg(DY_tau, "DY_tau")
    histo.add_bkg(DY_lep, "DY_lep")
    histo.add_bkg(Top, "Top_NJ")
    histo.add_bkg(VV, "VV_NJ")

    ax = histo.get_ax(xlabel="BDT Score for W+jets", lumi=lumi, ncol=2, fontsmall=True)

    ax.set_xlim(0,1)
    ax.set_ylabel("Counts")
    plt.savefig(os.path.join(model_dir, 'plots', f"W_score_distribution.pdf"))



if __name__ == "__main__":
    args = get_args()
    cfg = yaml.safe_load(open("../config/config_FFs.yaml"))
    # Load the correct config for the channel
    if args.channel == 'mt':
        print("Plotting for MuTau channel")
        cfg = cfg['mt']
        plot_score(cfg, "EVEN", 'mt')
        plot_score(cfg, "ODD", 'mt')
    elif args.channel == 'et':
        print("Plotting for ETau channel")
        cfg = cfg['et']
        plot_score(cfg, "EVEN", 'et')
        plot_score(cfg, "ODD", 'et')


