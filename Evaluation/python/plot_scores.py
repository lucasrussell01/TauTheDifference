import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import numpy as np
import yaml


import argparse

def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Classifier Evaluation")
    parser.add_argument('--channel', type=str, help="Channel to train", required=True)
    parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    return parser.parse_args()



def main(cfg, parity):

    lumi = 61.90

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


    # Set Bins
    bin_size = 0.1
    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_centre = bins[:-1]+ np.diff(bins)/2
    step_edges = np.append(bins,2*bins[-1]-bins[-2])

    # Weighting
    weight = "weight" # lumi*XS/Neff

    cat_dict = {'Tau': {'label': 0, 'pred_name': 'pred_0'},
                'Higgs': {'label': 1, 'pred_name': 'pred_1'},
                'Fake': {'label': 2, 'pred_name': 'pred_2'}
                }

    for cat_name, cat in cat_dict.items():

        # pick only events predicted to be the given category
        pred_cat_df = pred_df.loc[pred_df['pred_label'] == cat['label']]
        # print(pred_cat_df)

        # extract categories
        taus = pred_cat_df.loc[pred_cat_df['class_label'] == 0]
        fake = pred_cat_df[pred_cat_df['class_label'] == 2]
        ggH = pred_cat_df[pred_cat_df['class_label'] == 11]
        VBF = pred_cat_df[pred_cat_df['class_label'] == 12]
        total_bkg = pred_cat_df[(pred_cat_df['class_label'] == 0) | (pred_cat_df['class_label'] == 2)]


        print(f"Plotting distribution for {cat_name}")
        fig, ax = plt.subplots(figsize = (6,6))

        # Histograms of Tau and Background Classes
        tau_hist = np.histogram(taus[cat["pred_name"]], bins=bins, weights=taus[weight])[0]
        fake_hist = np.histogram(fake[cat["pred_name"]], bins=bins, weights=fake[weight])[0]
        ax.bar(bin_centre, tau_hist, width = bin_size, color = yellow, label = r"$Z\to\tau_h\tau_h$")
        ax.bar(bin_centre, fake_hist, width = bin_size, color = green, bottom = tau_hist, label = r"jet $\to \tau_h$ [QCD]")
        taus_step = np.append(np.insert(tau_hist,0,0.0),0.0)
        fake_step = np.append(np.insert(fake_hist,0,0.0),0.0) + taus_step
        ax.step(step_edges, taus_step, color='black', linewidth = 0.5)
        ax.step(step_edges, fake_step, color='black', linewidth = 0.5)

        # Outline histos of Higgs processes and Total Bkg
        ggH_counts, _, _ = ax.hist(ggH[cat["pred_name"]], bins=bins, histtype="step", color = red, linewidth = 2, label = r"ggH$\to\tau_h\tau_h$", weights=ggH[weight])
        VBF_counts, _, _ = ax.hist(VBF[cat["pred_name"]], bins=bins, histtype="step", color = blue, linewidth = 2, label = r"qqH$\to\tau_h\tau_h$", weights=VBF[weight])
        bkg_counts, _, _ = ax.hist(total_bkg[cat["pred_name"]], bins=bins, histtype="step", color = 'black', linewidth = 2, label = r"Total Background", weights=total_bkg[weight])

        # Labels etc
        ax.set_xlabel(rf"{cat_name} BDT Score")
        ax.set_ylabel(f"Events/{bin_size}  [{weight}]")
        ax.set_xlim(0.33, 1)
        ax.text(0.6, 1.02, rf"{lumi:.2f} fb$^{{-1}}$ (13.6 TeV)", fontsize=14, transform=ax.transAxes)
        ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        ax.text(0.15, 1.02, 'Work in Progress', fontsize=14, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
        ax.legend()
        ax.set_yscale('log')
        ax.set_ylim(1, 1e5)
        plt.savefig(os.path.join(model_dir, 'plots', f"{cat_name}_score.pdf"))

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
    elif args.channel == 'mt':
        print("Plotting for MuTau channel")
        cfg = cfg['mt']
    elif args.channel == 'et':
        print("Plotting for ETau channel")
        cfg = cfg['et']

    main(cfg, "ODD")
    main(cfg, "EVEN")

