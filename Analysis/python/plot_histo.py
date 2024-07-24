import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import pandas as pd

plt.style.use(hep.style.ROOT)
purple = (152/255, 152/255, 201/255)
yellow = (243/255,170/255,37/255)
blue = (2/255, 114/255, 187/255)
green = (159/255, 223/255, 132/255)
red = (203/255, 68/255, 10/255)

plt.rcParams.update({"font.size": 14})



# import merged SM file
merged_df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_ALL.parquet')

# extract categories
taus = merged_df.loc[merged_df['class_label'] == 0]
bkg = merged_df[merged_df['class_label'] == 2]
ggH = merged_df[merged_df['class_label'] == 11]
VBF = merged_df[merged_df['class_label'] == 12]

del merged_df


bin_size = 10
bins = np.arange(0, 300 + bin_size, bin_size)
bin_centre = bins[:-1]+ np.diff(bins)/2

step_edges = np.append(bins,2*bins[-1]-bins[-2])

# calculate counts
tau_hist = np.histogram(taus['m_vis'], bins=bins, weights=taus['weight'])[0]
bkg_hist = np.histogram(bkg['m_vis'], bins=bins, weights=bkg['weight'])[0]


fig, ax = plt.subplots(figsize = (6,6))
# histograms of non signal
ax.bar(bin_centre, tau_hist, width = bin_size, color = yellow, label = r"$Z\to\tau\tau$")
ax.bar(bin_centre, bkg_hist, width = bin_size, color = green, bottom = tau_hist, label = r"jet $\to \tau_h$ [QCD]")
# step outlines of the above
taus_step = np.append(np.insert(tau_hist,0,0.0),0.0)
bkg_step = np.append(np.insert(bkg_hist,0,0.0),0.0) + taus_step
ax.step(step_edges, taus_step, color='black', linewidth = 0.5)
ax.step(step_edges, bkg_step, color='black', linewidth = 0.5)
# outline histos of signal processes (reweighted)
ax.hist(ggH['m_vis'], bins=bins, weights=ggH['weight'], histtype="step", color = red, linewidth = 2, label = r"ggH$\to\tau\tau$")
ax.hist(VBF['m_vis'], bins=bins, weights=VBF['weight'], histtype="step", color = blue, linewidth = 2, label = r"qqH$\to\tau\tau$")
# labels etc
ax.set_xlabel(r"m$_{vis}$ (GeV)")
ax.set_ylabel(f"Weighted Events/{bin_size} GeV")
ax.set_xlim(0, 300)
ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
ax.legend()
plt.savefig(f"m_vis.pdf")