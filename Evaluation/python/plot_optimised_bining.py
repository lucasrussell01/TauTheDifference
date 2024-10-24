import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import numpy as np
import yaml
from statsmodels.stats.weightstats import DescrStatsW


def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def main(cfg, parity):

    lumi = 34.65
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
    total_bkg = pred_df[(pred_df['class_label'] == 0) | (pred_df['class_label'] == 2)]

    # Split into n bins with equal number of weighted signal
    n_bins = 5
    w_perc = DescrStatsW(higgs['pred_1'], weights=higgs['weight']).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins = np.concatenate([[0.33], np.array(w_perc), [1]])
    # print(f"Optimised bin edges: {bins}")

    bin_centre = bins[:-1]+ np.diff(bins)/2
    step_edges = np.append(bins,2*bins[-1]-bins[-2])

    weight = "weight" # lumi*XS/Neff

    print(f"Plotting distribution for Higgs")
    fig, ax = plt.subplots(figsize = (6,6))

    # Histograms of Tau and Background Classes
    tau_hist = np.histogram(taus['pred_1'], bins=bins, weights=taus[weight])[0]
    fake_hist = np.histogram(fake['pred_1'], bins=bins, weights=fake[weight])[0]
    ax.bar(bin_centre, tau_hist, width = np.diff(bins), color = yellow, label = r"$Z\to\tau_h\tau_h$")
    ax.bar(bin_centre, fake_hist, width = np.diff(bins), color = green, bottom = tau_hist, label = r"jet $\to \tau_h$ [QCD]")
    taus_step = np.append(np.insert(tau_hist,0,0.0),0.0)
    fake_step = np.append(np.insert(fake_hist,0,0.0),0.0) + taus_step
    ax.step(step_edges, taus_step, color='black', linewidth = 0.5)
    ax.step(step_edges, fake_step, color='black', linewidth = 0.5)

    # Outline histos of Higgs processes and Total Bkg
    ggH_counts, _, _ = ax.hist(ggH['pred_1'], bins=bins, histtype="step", color = red, linewidth = 2, label = r"ggH$\to\tau_h\tau_h$", weights=ggH[weight])
    VBF_counts, _, _ = ax.hist(VBF['pred_1'], bins=bins, histtype="step", color = blue, linewidth = 2, label = r"qqH$\to\tau_h\tau_h$", weights=VBF[weight])
    bkg_counts, _, _ = ax.hist(total_bkg['pred_1'], bins=bins, histtype="step", color = 'black', linewidth = 2, label = r"Total Background", weights=total_bkg[weight])

    # plot bin boundaries
    for i in range(1, n_bins):
        ax.axvline(x=bins[i], color='black', linestyle='--', linewidth = 1.3)

    sig_counts = ggH_counts + VBF_counts
    sig_AMS = AMS(sig_counts, bkg_counts)
    print("--------------------------------------------------------")
    # print(f"S/root(S+B) for the individual bins is: {sigs_SSB}")
    print(f"AMS for the individual bins is: {sig_AMS}")
    print(f"Overall AMS for {model_dir.split('/')[-1]}: {np.sqrt(np.sum(sig_AMS**2))}")

    # Labels etc
    ax.set_xlabel(rf"Higgs {cfg['model_type']} Score")
    ax.set_ylabel(f"Events (weighted)")
    ax.set_xlim(0.33, 1)
    ax.text(0.6, 1.02, rf"{lumi:.2f} fb$^{{-1}}$ (13.6 TeV)", fontsize=14, transform=ax.transAxes)
    ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.text(0.15, 1.02, 'Work in Progress', fontsize=14, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
    ax.text(0.685, 0.87, f'AMS: {np.sqrt(np.sum(sig_AMS**2)):.2f}\n{parity} Events\n{cfg["model_cut"]} VSjet Cut', fontsize=12, transform=ax.transAxes, fontfamily='sans-serif',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.legend(frameon=1, framealpha=1)
    ax.set_yscale('log')
    ax.set_ylim(1, 1e5)
    plt.savefig(os.path.join(model_dir, 'plots', f"Optimised_Higgs_score.pdf"))


if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/config.yaml"))
    main(cfg, "EVEN")
    main(cfg, "ODD")
