import pandas as pd
import matplotlib.pyplot as plt
import os
import mplhep as hep
import numpy as np

plt.style.use(hep.style.ROOT)
purple = (152/255, 152/255, 201/255)
yellow = (243/255,170/255,37/255)
blue = (2/255, 114/255, 187/255)
green = (159/255, 223/255, 132/255)
red = (203/255, 68/255, 10/255)

plt.rcParams.update({"font.size": 14})


# histogram the categories for different scores

model_dir = "../../Training/python/XGB_Models/BDTClassifier/model_2907"

# TODO: would be ideal to have separaete NN and other weights stored + maybe lumi scalings

pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))
class_label_counts = pred_df['class_label'].value_counts()
print(class_label_counts)

# extract categories
taus = pred_df.loc[pred_df['class_label'] == 0]
fake = pred_df[pred_df['class_label'] == 2]
ggH = pred_df[pred_df['class_label'] == 11]
VBF = pred_df[pred_df['class_label'] == 12]
total_bkg = pred_df[(pred_df['class_label'] == 0) | (pred_df['class_label'] == 2)]

# Set Bins
bin_size = 0.1
bins = np.arange(0, 1 + bin_size, bin_size)
bin_centre = bins[:-1]+ np.diff(bins)/2
step_edges = np.append(bins,2*bins[-1]-bins[-2])

# TODO: Plot with no weights (eg lumi/XS) for now -> because then Higgs become  tiny?

cat_dict = {'pred_0': 'Tau', 'pred_1': 'Higgs', 'pred_2': 'Fake'}

for cat, cat_label in cat_dict.items():
    print(f"Plotting distribution for {cat_label}")
    fig, ax = plt.subplots(figsize = (6,6))
    
    # Histograms of Tau and Background Classes
    tau_hist = np.histogram(taus[cat], bins=bins)[0]#, weights=taus['weight'])[0]
    fake_hist = np.histogram(fake[cat], bins=bins)[0]#, weights=fake['weight'])[0]
    ax.bar(bin_centre, tau_hist, width = bin_size, color = yellow, label = r"$Z\to\tau_h\tau_h$")
    ax.bar(bin_centre, fake_hist, width = bin_size, color = green, bottom = tau_hist, label = r"jet $\to \tau_h$ [QCD]")
    taus_step = np.append(np.insert(tau_hist,0,0.0),0.0)
    fake_step = np.append(np.insert(fake_hist,0,0.0),0.0) + taus_step
    ax.step(step_edges, taus_step, color='black', linewidth = 0.5)
    ax.step(step_edges, fake_step, color='black', linewidth = 0.5)
    
    # Outline histos of Higgs processes and Total Bkg
    ax.hist(ggH[cat], bins=bins, histtype="step", color = red, linewidth = 2, label = r"ggH$\to\tau_h\tau_h$")#, weights=ggH['weight']
    ax.hist(VBF[cat], bins=bins, histtype="step", color = blue, linewidth = 2, label = r"qqH$\to\tau_h\tau_h$")#, weights=VBF['weight']
    ax.hist(total_bkg[cat], bins=bins, histtype="step", color = 'black', linewidth = 2, label = r"Total Background")#, weights=total_bkg['weight']

    # Labels etc
    ax.set_xlabel(rf"{cat_label} BDT Score")
    ax.set_ylabel(f"Events/{bin_size}  (Unweighted)")
    ax.set_xlim(0, 1)
    ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
    ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(1, 1e6)
    plt.savefig(os.path.join(model_dir, f"{cat_label}_score.pdf"))




