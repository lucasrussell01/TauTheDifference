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


def SrootSB(S,B):
    # signal over root signal + background
    return S/np.sqrt(S+B)

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams


# histogram the categories for different scores

model_dir = "../../Training/python/XGB_Models/BDTClassifier/model_2907"

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

# decide bins with flat regions in signal
signal_scores = higgs['pred_1']

# find number of signal (weighted) to have per bin
n_sig = len(signal_scores)
n_bins = 5
sum_w = np.sum(higgs['weight'])
target_per_bin = sum_w/n_bins

# scan bins until get to target amount of signal weight
higgs = higgs.sort_values(by='pred_1').reset_index(drop=True)

sum_w_bin = 0 # track sum of weights in a bin

# ugly, but iterate through the signal df, and check if the cumulative weight
# is > thresh to make a new bin
bins = [0.33]
for i in range(n_sig-1):
    sum_w_bin += higgs['weight'][i]
    if sum_w_bin > target_per_bin:
        bin_edge = (higgs['pred_1'][i] + higgs['pred_1'][i+1])/2
        bins.append(bin_edge)
        sum_w_bin = 0
bins.append(1)


print(f"Optimised bining is: {bins}")


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

# Labels etc
ax.set_xlabel(rf"Higgs BDT Score")
ax.set_ylabel(f"Events (weighted)")
ax.set_xlim(0.33, 1)
ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
ax.legend(frameon=1, framealpha=1)
ax.set_yscale('log')
ax.set_ylim(1, 1e5)
plt.savefig(os.path.join(model_dir, f"Optimised_Higgs_score.pdf"))


sig_counts = ggH_counts + VBF_counts
sigs_SSB = np.zeros(n_bins)
sigs_AMS = np.zeros(n_bins)
for b in range(n_bins):
    sig_SSB = SrootSB(sig_counts[b], bkg_counts[b])
    sig_AMS = AMS(sig_counts[b], bkg_counts[b])
    sigs_SSB[b] = sig_SSB
    sigs_AMS[b] = sig_AMS

print("--------------------------------------------------------")
print(f"S/root(S+B) for the individual bins is: {sigs_SSB}")
print(f"AMS for the individual bins is: {sigs_AMS}")

overall_sig_SSB = np.sqrt(np.sum(sigs_SSB**2))
overall_sig_AMS = np.sqrt(np.sum(sigs_AMS**2))
print(f"Overall S/root(S+B) (sum in quad): {overall_sig_SSB}")
print(f"Overall AMS (sum in quad): {overall_sig_AMS}")
