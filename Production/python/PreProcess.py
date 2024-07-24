import numpy as np
import pandas as pd
from glob import glob
import os
import json
from tqdm import tqdm
from run_info import concat_json

# Select same sign of opposite sign pairs from parquet files

base_dir = '/vols/cms/lcr119/offline/HiggsCP/data/raw/2022/tt'

samples = {
    'DYto2L_M-50_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_madgraphMLM_ext1': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_1J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_2J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_3J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'DYto2L_M-50_4J_madgraphMLM': {'data': False, 'label': 0, 'x_sec': 6282.6, 'n_eff': 145286646, 'process': 'DY'},
    'GluGluHToTauTau_M125': {'data': False, 'label': 11, 'x_sec': 3.276, 'n_eff': 295692, 'process': 'Higgs'},
    'VBFHToTauTau_M125': {'data': False, 'label': 12, 'x_sec': 0.2558, 'n_eff': 298955, 'process': 'Higgs'},
    'Tau_Run2022C': {'data': True, 'label': 2, 'x_sec': -1, 'n_eff': -1, 'process': 'QCD'},
    'Tau_Run2022D': {'data': True, 'label': 2, 'x_sec': -1, 'n_eff': -1, 'process': 'QCD'}
          }

out_dir = '/vols/cms/lcr119/offline/HiggsCP/data/processed/2022/tt'


def proc_weight(x_sec, n_eff, lumi = 8077):
    # 8077 is 2022 lumi 
    return (x_sec * lumi) / n_eff


# Features to keep
features = ['pt_1', 'pt_2', 'eta_1', 'eta_2', 'phi_1', 'phi_2', 'dR', 'pt_tt', 'pt_tt_met',
            'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'met', 'met_phi', 'met_dphi_1', 'met_dphi_2', 
            'dphi', 'm_vis', 'pt_vis', 'n_jets', 'n_bjets', 'mjj', 'jdeta', 'sjdphi', 'dijetpt', 
            'jpt_1', 'jpt_2', 'jeta_1', 'jeta_2', 'jphi_1', 'jphi_2', 
            'weight', 'class_label']
print("Features to store:", features)

            

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
        # df['w_process'] = proc_factor # store process factor separately
    # Select sign of the pair
    df = df[df['os'] == opposite_sign]
    # Set a class label
    df["class_label"] = options['label']
    # Discard unwanted features
    df = df[features]
    # Print summary of selection
    num_events_after = len(df)
    print(f"Number of events before: {n_before}\nNumber of events after [os = {opposite_sign}]: {num_events_after} ({(num_events_after / n_before) * 100:.2f}% kept)\n")
    # Save processed dataframe
    out_path = os.path.join(out_dir, sample)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    df.to_parquet(os.path.join(out_path, "merged_filtered.parquet"), engine="pyarrow")

