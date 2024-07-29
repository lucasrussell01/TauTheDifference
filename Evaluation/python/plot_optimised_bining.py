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

# extract categories
taus = pred_df.loc[pred_df['class_label'] == 0]
fake = pred_df[pred_df['class_label'] == 2]
higgs = pred_df[(pred_df['class_label'] == 11) | (pred_df['class_label'] == 12)]
ggH = pred_df[pred_df['class_label'] == 11]
VBF = pred_df[pred_df['class_label'] == 12]
total_bkg = pred_df[(pred_df['class_label'] == 0) | (pred_df['class_label'] == 2)]

# decide bins with flat regions in signal
signal_scores = higgs['pred_1']

# find number of signal to have per bin
n_sig = len(signal_scores)
n_bins = 5
n_sig_per_bin = int(n_sig/n_bins)
print(f"Splitting into {n_bins} bins, corresponding to {n_sig_per_bin} signal events per bin")

bins = [0] # start at 0
# find cuts on signal scores than lead to n_sig_per_bin
signal_scores = np.sort(signal_scores)
for n in range(1, n_bins): # don't start at first bin (should be zero)
    index = n*n_sig_per_bin
    # print(f"At index {index}/{n_sig} the score is {signal_scores[index]}")
    bins.append(signal_scores[index])
bins.append(1) # end at 1

print(f"Optimised bining is: {bins}")


bin_centre = bins[:-1]+ np.diff(bins)/2
print(bin_centre)
step_edges = np.append(bins,2*bins[-1]-bins[-2])
print(step_edges)

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
ax.hist(ggH['pred_1'], bins=bins, histtype="step", color = red, linewidth = 2, label = r"ggH$\to\tau_h\tau_h$", weights=ggH[weight])
ax.hist(VBF['pred_1'], bins=bins, histtype="step", color = blue, linewidth = 2, label = r"qqH$\to\tau_h\tau_h$", weights=VBF[weight])
ax.hist(total_bkg['pred_1'], bins=bins, histtype="step", color = 'black', linewidth = 2, label = r"Total Background", weights=total_bkg[weight])

# plot bin boundaries
for i in range(1, n_bins):
    ax.axvline(x=bins[i], color='black', linestyle='--', linewidth = 1.3)

# Labels etc
ax.set_xlabel(rf"Higgs BDT Score")
ax.set_ylabel(f"Events (weighted)")
ax.set_xlim(0, 1)
ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
ax.legend()
ax.set_yscale('log')
ax.set_ylim(1, 1e6)
plt.savefig(os.path.join(model_dir, f"Optimised_Higgs_score.pdf"))

