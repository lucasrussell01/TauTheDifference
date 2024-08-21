import numpy as np
import pandas as pd
from glob import glob
import os
import json
from tqdm import tqdm

# Select same sign of opposite sign pairs from parquet files

base_dir = '/vols/cms/lcr119/offline/HiggsCP/HiggsDNA/output/Run3_2022/tt'

samples = {
    'DYto2L_M-50_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_madgraphMLM_ext1': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_1J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_2J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_3J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_4J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'Tau_Run2022C': {'data': True, 'label': 2, 'x_sec': -1, 'n_eff': -1, 'process': 'QCD'},
    'Tau_Run2022D': {'data': True, 'label': 2, 'x_sec': -1, 'n_eff': -1, 'process': 'QCD'},
    'GluGluHTo2Tau_UncorrelatedDecay_SM_Filtered_ProdAndDecay':  {'data': False, 'label': 11, 'x_sec': 3.276, 'n_eff': 20313722, 'process': 'Higgs', 'filter_eff': 3.847e-01},
    'GluGluHTo2Tau_UncorrelatedDecay_MM_Filtered_ProdAndDecay':  {'data': False, 'label': 11, 'x_sec': 3.276, 'n_eff': 20313722, 'process': 'Higgs', 'filter_eff': 3.848e-01},
    'GluGluHTo2Tau_UncorrelatedDecay_CPodd_Filtered_ProdAndDecay':  {'data': False, 'label': 11, 'x_sec': 3.276, 'n_eff': 20313722, 'process': 'Higgs', 'filter_eff': 3.848e-01},
    'VBFHToTauTau_UncorrelatedDecay_Filtered':  {'data': False, 'label': 12, 'x_sec': 0.2558, 'n_eff': 5082505, 'process': 'Higgs', 'filter_eff': 4.091e-01}
    # 'GluGluHToTauTau_M125': {'data': False, 'label': 11, 'x_sec': 3.276, 'n_eff': 295692, 'process': 'Higgs'},
    # 'VBFHToTauTau_M125': {'data': False, 'label': 12, 'x_sec': 0.2558, 'n_eff': 298955, 'process': 'Higgs'},
          }


out_dir = '/vols/cms/lcr119/offline/HiggsCP/data/processed/2022/tt'


def proc_weight(x_sec, n_eff, lumi = 8077):
    # 8077 is 2022 lumi 
    return (x_sec * lumi) / n_eff

def cp_weight(df):
    cp_weight = 0.5*(df['LHEReweightingWeight_SM'] + df['LHEReweightingWeight_PS'])
    df['weight'] *= cp_weight
    # drop negative weights - combi can be negative (rarely)
    df = df[df['weight'] > 0]
    print("CP reweighting applied (-ve weights dropped)")
    return df


# Features to keep
features = ['pt_1', 'pt_2', 'eta_1', 'eta_2', 'phi_1', 'phi_2', 'dR', 'pt_tt', 'pt_tt_met',
            'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'met_pt', 'met_dphi_1', 'met_dphi_2',
            'dphi', 'm_vis', 'pt_vis', 'n_jets', 'n_bjets', 'mjj', 'jdeta', 'dijetpt',
            'jpt_1', 'jpt_2', 'jeta_1', 'jeta_2',
            'FastMTT_Mass', 'svfitMass', 'svfitMass_err',
            'weight', 'class_label']

print("Features to store:", features, "\n")


def cuts(df):
    print("Applying cuts")
    # VSjet cuts
    df = df[df['idDeepTau2018v2p5VSjet_1'] >= 5] # medium
    df = df[df['idDeepTau2018v2p5VSjet_2'] >= 5]
    # VSe cuts
    df = df[df['idDeepTau2018v2p5VSe_1'] >= 2] # vvloose
    df = df[df['idDeepTau2018v2p5VSe_2'] >= 2]
    # VSmu cuts
    df = df[df['idDeepTau2018v2p5VSmu_1'] >= 1] # vloose
    df = df[df['idDeepTau2018v2p5VSmu_2'] >= 1]
    return df


for sample, options in samples.items():
    # Open merged parquet file from HiggsDNA output
    file_path = os.path.join(base_dir, sample, 'nominal/merged.parquet')
    print(f"Processing [{sample}], label: {options['label']}")
    df = pd.read_parquet(file_path)
    n_before = len(df)
    if options['data']:
        opposite_sign = False # same sign data-driven QCD
        # NB: could eventually add extrapolation factor ~1
    elif not options['data']:
        opposite_sign = True # opposite sign for genuine ditau pairs
        # scale MC by luminosity and cross section
        proc_factor = proc_weight(options['x_sec'], options['n_eff'])
        print(f"Luminosity and Cross Section scaling applied: {proc_factor}")
        df['weight'] *= proc_factor # update central weight
    # CP sample adjustments:
    if 'Filtered' in sample:
        df['weight'] *= options['filter_eff']
        print(f"Filter efficiency applied: {options['filter_eff']}")
    if 'ProdAndDecay' in sample:
        df = cp_weight(df) # LHE reweighting
    # apply cuts
    df = cuts(df)
    num_events_after_cuts = len(df)
    # Select sign of the pair
    df = df[df['os'] == opposite_sign]
    # Set a class label
    df["class_label"] = options['label']
    # Discard unwanted features
    df = df[features]
    # Print summary of selection
    num_events_after_os = len(df)
    print(f"Number of events before: {n_before}")
    print(f"Number of events after selection: {num_events_after_cuts}: ({(num_events_after_cuts / n_before) * 100:.2f}% kept)")
    print(f"Number of events after [os = {opposite_sign}]: {num_events_after_os} ({(num_events_after_os / n_before) * 100:.2f}% kept)\n")
    # Save processed dataframe
    out_path = os.path.join(out_dir, sample)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_parquet(os.path.join(out_path, "merged_filtered.parquet"), engine="pyarrow")

