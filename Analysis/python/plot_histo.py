import numpy as np
import matplotlib.pyplot as plt
import json
import mplhep as hep
import pandas as pd
import argparse
import os
from plot_utils import stacked_histogram

plt.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 14})

# example usage: python plot_histo.py --var='m_vis' --label='m$_{vis}$ (GeV)'

lumi_22 = 7.9804
lumi_22EE = 26.6717
lumi_23 = 17.794
lumi_23BPix = 9.451

def get_args():
    parser = argparse.ArgumentParser(description="Plot histogram for a variable of choice")
    parser.add_argument('--channel', type=str, help="Channel to plot", required=True)
    parser.add_argument('--var', type=str, help="Variable to plot in df", required=True)
    parser.add_argument('--era', type=str, help="Era to plot", required=False, default='run3')
    parser.add_argument('--xmin', type=float, help="Min x to plot", required=False, default=0)
    parser.add_argument('--xmax', type=float, help="Max x to plot", required=False, default=350)
    parser.add_argument('--ymax', type=float, help="Max y to plot", required=False)
    parser.add_argument('--nbins', type=int, help="Number of bins", required=False, default=70)
    parser.add_argument('--label', type=str, help="Label name for the variable", required=False)
    parser.add_argument('--weight', type=str, help="name of weight column to use", required=False, default='weight')
    return parser.parse_args()

args = get_args()
if args.label is None:
    args.label = args.var

channel = args.channel

# File to draw from
base_path = '/vols/cms/lcr119/offline/HiggsCP/data/earlyrun3/ShuffleMerge'
file = os.path.join(base_path, channel, 'ShuffleMerge_ALL.parquet')


# import merged SM file
df = pd.read_parquet(file)#
if args.era == '2022':
    merged_df = df[df['era']==1]
    lumi = lumi_22
elif args.era == '2022EE':
    merged_df = df[df['era']==2]
    lumi = lumi_22EE
elif args.era == '2023':
    merged_df = df[df['era']==3]
    lumi = lumi_23
elif args.era == '2023BPix':
    merged_df = df[df['era']==4]
    lumi = lumi_23BPix
else:
    merged_df = df
    lumi = lumi_22 + lumi_22EE + lumi_23 + lumi_23BPix
    print(f"Using all available eras (full 2022 + 23)")



fig, ax = plt.subplots(figsize = (6,6))

if channel == 'tt':

    # Initialise plotting class
    bins = np.linspace(args.xmin, args.xmax, num=args.nbins+1)
    histo = stacked_histogram(args.var, ax, bins)
    # extract categories from dataframe
    # Genuine
    taus = merged_df.loc[merged_df['process_id'] == 11]
    # Fake
    bkg = merged_df[merged_df['process_id'] == 0]
    # Signal
    ggH = merged_df[merged_df['process_id'] == 100]
    VBF = merged_df[merged_df['process_id'] == 101]
    del merged_df
    # Add background processes
    histo.add_bkg(taus, "DY")
    histo.add_bkg(bkg, "Jet_Fakes")
    # Add signal processes
    histo.add_signal(ggH, "ggH")
    histo.add_signal(VBF, "VBF")

elif channel == 'mt':

    # Initialise plotting class
    bins = np.linspace(args.xmin, args.xmax, num=args.nbins+1) # wider binning
    histo = stacked_histogram(args.var, ax, bins)

    # extract categories
    # genuine taus
    DY_tau = merged_df.loc[merged_df['process_id'] == 11]
    other_tau = merged_df.loc[(merged_df['process_id'] == 21) | (merged_df['process_id'] == 31) | (merged_df['process_id'] == 51)] # TT, ST, VV
    # lepton fakes
    DY_lep = merged_df.loc[merged_df['process_id'] == 12]
    # jet fakes
    EW = merged_df.loc[(merged_df['process_id'] == 43) | (merged_df['process_id'] == 53)]
    QCD = merged_df.loc[merged_df['process_id'] == 0]
    Top_jet = merged_df.loc[(merged_df['process_id'] == 23) | (merged_df['process_id'] == 33)]
    other_jet = merged_df.loc[merged_df['process_id'] == 13]
    # signal
    ggH = merged_df[merged_df['process_id'] == 100]
    VBF = merged_df[merged_df['process_id'] == 101]
    del merged_df

    # Add fake processes
    histo.add_bkg(Top_jet, "Top_jet", weight=args.weight)
    histo.add_bkg(QCD, "QCD", weight=args.weight)
    histo.add_bkg(EW, "EW", weight=args.weight)
    histo.add_bkg(other_jet, "OtherFake", weight=args.weight)
    histo.add_bkg(DY_lep, "DY_lep", weight=args.weight)
    # genuine backgrounds
    histo.add_bkg(DY_tau, "DY", weight=args.weight)
    histo.add_bkg(other_tau, "OtherGenuine", weight=args.weight)
    # Add signal processes
    histo.add_signal(ggH, "ggH", weight=args.weight)
    histo.add_signal(VBF, "VBF", weight=args.weight)


# Get the axes
ax = histo.get_ax(xlabel=args.label, lumi=lumi, unit='GeV', channel=channel)
# Set the limits
ax.set_xlim(args.xmin, args.xmax)
if args.ymax is not None:
    ax.set_ylim(-100, args.ymax)
else:
    ax.set_ylim(-1e-3*histo.get_max(), 1.15*histo.get_max())
# Figure Saving
fname = f"figs/{args.var}_{args.era}_{channel}_{args.weight}.pdf"
plt.tight_layout()
plt.savefig(fname)
print(f"Plotted {args.var} to {fname}")





