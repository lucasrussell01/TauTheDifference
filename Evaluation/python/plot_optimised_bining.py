import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import numpy as np
import yaml
from statsmodels.stats.weightstats import DescrStatsW
from plot_utils import stacked_histogram

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def main(cfg, parity):

    lumi = 61.90
    # Plotting style
    plt.style.use(hep.style.ROOT)
    purple = (152/255, 152/255, 201/255)
    yellow = (243/255,170/255,37/255)
    blue = (2/255, 114/255, 187/255)
    green = (159/255, 223/255, 132/255)
    red = (203/255, 68/255, 10/255)
    plt.rcParams.update({"font.size": 14})

    # histogram the categories for different scores

    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)

    pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))
    class_label_counts = pred_df['class_label'].value_counts()

    # only events classified as Higgs
    pred_df = pred_df[pred_df['pred_label'] == 1]

    # extract categories
    taus = pred_df.loc[pred_df['class_label'] == 0]
    fake = pred_df[pred_df['class_label'] == 2]
    higgs = pred_df[(pred_df['class_label'] == 11) | (pred_df['class_label'] == 12)]
    ggH = pred_df[pred_df['class_label'] == 11]
    VBF = pred_df[pred_df['class_label'] == 12]

    # Split into n bins with equal number of weighted signal
    n_bins = 5
    w_perc = DescrStatsW(higgs['pred_1'], weights=higgs['weight']).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins = np.concatenate([[0.33], np.array(w_perc), [1]])
    # print(f"Optimised bin edges: {bins}")

    weight = "weight" # lumi*XS/Neff

    print(f"Plotting distribution for Higgs")
    fig, ax = plt.subplots(figsize = (6,6))

    histo = stacked_histogram("pred_1", ax, bins)
    # Tau and Background Classes
    histo.add_bkg(taus, "DY")
    histo.add_bkg(fake, "Jet_Fakes")
    histo.add_total_bkg()
    # Signal Processes
    histo.add_signal(ggH, "ggH")
    histo.add_signal(VBF, "VBF")

    # Get the axes
    ax = histo.get_ax(xlabel=rf"Higgs {cfg['model_type']} Score", lumi=lumi)

    # plot bin boundaries
    for i in range(1, n_bins):
        ax.axvline(x=bins[i], color='black', linestyle='--', linewidth = 1.3)

    # Get counts for AMS
    sig_counts, bkg_counts = histo.get_counts()
    print(sig_counts, bkg_counts)
    sig_AMS = AMS(sig_counts, bkg_counts)
    print("--------------------------------------------------------")
    print(f"AMS for the individual bins is: {sig_AMS}")
    print(f"Overall AMS for {model_dir.split('/')[-1]}: {np.sqrt(np.sum(sig_AMS**2))}")

    # Labels etc
    ax.set_xlim(0.33, 1)
    ax.text(0.685, 0.87, f'AMS: {np.sqrt(np.sum(sig_AMS**2)):.2f}\n{parity} Events\n{cfg["model_cut"]} VSjet Cut', fontsize=12, transform=ax.transAxes, fontfamily='sans-serif',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_yscale('log')
    ax.set_ylim(1, 1e5)
    plt.savefig(os.path.join(model_dir, 'plots', f"Optimised_Higgs_score_TEST.pdf"))


if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/config.yaml"))
    main(cfg, "EVEN")
    main(cfg, "ODD")
