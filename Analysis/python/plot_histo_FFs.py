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
    parser.add_argument('--nbins', type=int, help="Number of bins", required=False, default=35)
    parser.add_argument('--label', type=str, help="Label name for the variable", required=False)
    parser.add_argument('--weight', type=str, help="name of weight column to use", required=False, default='weight')
    parser.add_argument('--signal', action='store_true', help="Signal Only")
    return parser.parse_args()

args = get_args()

for var in ['pt_1', 'pt_2', 'abs_eta_1', 'dR', 'dphi', 'pt_tt', 'm_vis', 'pt_vis', 'FastMTT_mass', 'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'jpt_1', 'jpt_2', 'jeta_1', 'jeta_2', 'mjj', 'jdeta', 'dijetpt', 'n_jets']:
    args.var = var
    print(f"\n Plotting variable: {args.var}", '*'*50)


    args.label = args.var


    channel = args.channel

    dir = 'FF_BDT'
    # File to draw from
    base_path = f'/vols/cms/lcr119/offline/HiggsCP/data/{dir}/ShuffleMerge'
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


    if channel == 'mt' or channel == 'et' or channel == 'tt':

        # Initialise plotting class
        bins = np.linspace(args.xmin, args.xmax, num=args.nbins+1) # wider binning
        histo = stacked_histogram(args.var, ax, bins)

        # extract categories
        # genuine taus
        DY_tau = merged_df.loc[merged_df['process_id'] == 11]
        print(f'DY tau events: {np.sum(DY_tau[args.weight])}')
        DY_lep = merged_df.loc[merged_df['process_id'] == 12]
        print(f'DY lep events: {np.sum(DY_lep[args.weight])}')
        TT_tau = merged_df.loc[(merged_df['process_id'] == 21)] # TT, ST, VV
        print(f'TT tau events: {np.sum(TT_tau[args.weight])}')
        VV_tau = merged_df.loc[((merged_df['process_id'] == 51) | (merged_df["process_id"] == 31))]
        print(f'VV tau events: {np.sum(VV_tau[args.weight])}')
        # jet fakes
        Wjets = merged_df.loc[(merged_df['process_id'] == 43) ]
        print(f'W+jets events: {np.sum(Wjets[args.weight])}')
        del merged_df

        if not args.signal: # not just signal
            # Add fake processes
            histo.add_bkg(Wjets, "WJets", weight=args.weight)
            # genuine backgrounds
            histo.add_bkg(DY_tau, "DY_tau", weight=args.weight)
            histo.add_bkg(TT_tau, "Top_tau", weight=args.weight)
            histo.add_bkg(VV_tau, "VVST_tau", weight=args.weight)
            histo.add_bkg(DY_lep, "DY_lep", weight=args.weight)


        # plot a cut
        # ax.axvline(x=0, color='red', linestyle='--', label='default cut')

    print(f'boooo {args.label}')
    # Get the axes
    ax = histo.get_ax(xlabel=args.label, lumi=lumi, unit='', channel=channel)
    # Set the limits
    ax.set_xlim(args.xmin, args.xmax)
    if args.ymax is not None:
        ax.set_ylim(-100, args.ymax)
    else:
        ax.set_ylim(-1e-3*histo.get_max(), 1.15*histo.get_max())
    # Figure Saving
    if not args.signal:
        fname = f"{dir}/{args.era}_{channel}_{args.var}_{args.weight}.pdf"
    else:
        fname = f"{dir}/SIGNAL_{args.era}_{channel}_{args.var}_{args.weight}.pdf"
    plt.tight_layout()
    plt.savefig(fname)
    print(f"Plotted {args.var} to {fname}")





