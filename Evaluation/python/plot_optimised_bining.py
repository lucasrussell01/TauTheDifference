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
    parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    return parser.parse_args()

# Plotting style
plt.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 14})

# Luminosity
lumi = 61.90

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def plot_tt(cfg, parity):
    # Load the model predictions
    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)
    pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))
    process_id_counts = pred_df['process_id'].value_counts()

    # Select events classified as Higgs
    pred_df = pred_df[pred_df['pred_label'] == 1]

    # Split into individual classes (by process ID)
    taus = pred_df.loc[pred_df['process_id'] == 11]
    fake = pred_df[pred_df['process_id'] == 0]
    ggH = pred_df[pred_df['process_id'] == 100]
    VBF = pred_df[pred_df['process_id'] == 101]
    # All Higgs for binning
    higgs = pred_df[(pred_df['process_id'] == 100) | (pred_df['process_id'] == 101)]

    # Split into n bins with equal number of weighted signal
    n_bins = 5
    w_perc = DescrStatsW(higgs['pred_1'], weights=higgs['weight']).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins = np.concatenate([[0.33], np.array(w_perc), [1]])

    # Plot the optimised distribution
    print(f"Plotting optimised distribution for Higgs")

    fig, ax = plt.subplots(figsize = (6,6))

    # Stacked histogram
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
    # print(sig_counts, bkg_counts)
    sig_AMS = AMS(sig_counts, bkg_counts)
    print("--------------------------------------------------------")
    print(f"AMS for the individual bins is: {sig_AMS}")
    print(f"Overall AMS for {model_dir.split('/')[-1]}: {np.sqrt(np.sum(sig_AMS**2))}")

    # Labels and AMS display
    ax.set_xlim(0.33, 1)
    box_info = rf"""AMS: {np.sqrt(np.sum(sig_AMS**2)):.2f}
{parity} Events
{cfg["model_cut"]} VSjet Cut
$\tau_h\tau_h$ channel"""

    ax.text(0.685, 0.87, box_info, fontsize=12, transform=ax.transAxes, fontfamily='sans-serif',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_yscale('log')
    ax.set_ylim(1, 1e5)
    plt.savefig(os.path.join(model_dir, 'plots', f"Optimised_Higgs_score.pdf"))


def plot_mt(cfg, parity):
    # Load the model predictions
    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)
    pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))
    process_id_counts = pred_df['process_id'].value_counts()

    # Select events classified as Higgs
    pred_df = pred_df[pred_df['pred_label'] == 1]

    # Split into individual classes (by process ID)
    # genuine taus
    DY_tau = pred_df.loc[pred_df['process_id'] == 11]
    other_tau = pred_df.loc[(pred_df['process_id'] == 21) | (pred_df['process_id'] == 31) | (pred_df['process_id'] == 51)] # TT, ST, VV
    # lepton fakes
    DY_lep = pred_df.loc[pred_df['process_id'] == 12]
    # jet fakes
    EW = pred_df.loc[(pred_df['process_id'] == 43) | (pred_df['process_id'] == 53)]
    QCD = pred_df.loc[pred_df['process_id'] == 0]
    Top_jet = pred_df.loc[(pred_df['process_id'] == 23) | (pred_df['process_id'] == 33)]
    other_jet = pred_df.loc[pred_df['process_id'] == 13]
    # signal
    ggH = pred_df[pred_df['process_id'] == 100]
    VBF = pred_df[pred_df['process_id'] == 101]
    # All Higgs for binning
    higgs = pred_df[(pred_df['process_id'] == 100) | (pred_df['process_id'] == 101)]

    # Split into n bins with equal number of weighted signal
    n_bins = 5
    w_perc = DescrStatsW(higgs['pred_1'], weights=higgs['weight']).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins = np.concatenate([[0.33], np.array(w_perc), [1]])

    # Plot the optimised distribution
    print(f"Plotting optimised distribution for Higgs")
    fig, ax = plt.subplots(figsize = (6,6))
    # Stacked histogram
    histo = stacked_histogram("pred_1", ax, bins)

    # Add fake processes
    histo.add_bkg(Top_jet, "Top_jet")
    histo.add_bkg(QCD, "QCD")
    histo.add_bkg(EW, "EW")
    histo.add_bkg(other_jet, "OtherFake")
    histo.add_bkg(DY_lep, "DY_lep")
    # genuine backgrounds
    histo.add_bkg(DY_tau, "DY")
    histo.add_bkg(other_tau, "OtherGenuine")
    # Total background outline
    histo.add_total_bkg()
    # Add signal processes
    histo.add_signal(ggH, "ggH")
    histo.add_signal(VBF, "VBF")

    # Get the axes
    ax = histo.get_ax(xlabel=rf"Higgs {cfg['model_type']} Score", lumi=lumi, ncol=2, fontsmall=True)

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

    # Labels and AMS display
    ax.set_xlim(0.33, 1)
    box_info = rf"""AMS: {np.sqrt(np.sum(sig_AMS**2)):.2f}
{parity} Events
$\mu\tau_h$ channel"""
    ax.text(0.75, 0.87, box_info, fontsize=12, transform=ax.transAxes, fontfamily='sans-serif',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_yscale('log')
    ax.set_ylim(1, 100*histo.get_max())
    plt.savefig(os.path.join(model_dir, 'plots', f"Optimised_Higgs_score.pdf"))



if __name__ == "__main__":
    args = get_args()
    cfg = yaml.safe_load(open("../config/config.yaml"))
    # Load the correct config for the channel (and vsjet cut)
    if args.channel == 'tt': # Fully hadronic has different vsjet cuts
        if args.cut == "tight":
            print("Plotting for tt channel (TIGHT Vsjet cut)")
            cfg = cfg['tt_tight']
        elif args.cut == "vtight":
            print("Plotting for tt channel (VTIGHT Vsjet cut)")
            cfg = cfg['tt_vtight']
        else: # use medium by default
            print("Plotting for tt channel (MEDIUM Vsjet cut)")
            cfg = cfg['tt_medium']
        plot_tt(cfg, "EVEN")
        plot_tt(cfg, "ODD")
    elif args.channel == 'mt':
        print("Plotting for MuTau channel")
        cfg = cfg['mt']
        plot_mt(cfg, "EVEN")
        plot_mt(cfg, "ODD")


