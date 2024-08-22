import numpy as np
import pandas as pd
from glob import glob
import os
import json
import yaml

# Process HiggsDNA outputs for classifier training


def process_weight(x_sec, n_eff, lumi = 8077):
    # 8077 is 2022 lumi 
    return (x_sec * lumi) / n_eff

def cp_weight(df):
    cp_weight = 0.5*(df['LHEReweightingWeight_SM'] + df['LHEReweightingWeight_PS'])
    df['weight'] *= cp_weight
    # drop negative weights - combi can be negative (rarely)
    df = df[df['weight'] > 0]
    print("CP reweighting applied (-ve weights dropped)")
    return df

def cuts(df, os, sel_cfg):
    print("Applying cuts")
    # VSjet cuts
    print(f"VSjet cuts - tau1: {sel_cfg['vsjet_1']}, tau2: {sel_cfg['vsjet_2']}")
    df = df[df['idDeepTau2018v2p5VSjet_1'] >= sel_cfg['vsjet_1']]
    df = df[df['idDeepTau2018v2p5VSjet_2'] >= sel_cfg['vsjet_2']]
    # VSe cuts
    print(f"VSe cuts - tau1: {sel_cfg['vse_1']}, tau2: {sel_cfg['vse_2']}")
    df = df[df['idDeepTau2018v2p5VSe_1'] >= sel_cfg['vse_1']]
    df = df[df['idDeepTau2018v2p5VSe_2'] >= sel_cfg['vse_2']]
    # VSmu cuts
    print(f"VSmu cuts - tau1: {sel_cfg['vsmu_1']}, tau2: {sel_cfg['vsmu_2']}")
    df = df[df['idDeepTau2018v2p5VSmu_1'] >= sel_cfg['vsmu_1']]
    df = df[df['idDeepTau2018v2p5VSmu_2'] >= sel_cfg['vsmu_2']]
    # Opposite sign cut
    print(f"Opposite sign requirement: {os}")
    df = df[df['os'] == os]
    return df

def process_samples(cfg):
    # Preprocessing for NN training, for all samples specified
    for sample, options in cfg['Datasets'].items():
        # Open merged parquet file from HiggsDNA output
        file_path = os.path.join(cfg['Paths']['input'], sample, 'nominal/merged.parquet')
        print(f"Processing [{sample}]")
        df = pd.read_parquet(file_path)

        # Determine relevant weights
        if not options['data']:
            opposite_sign = True # same sign data-driven QCD
            process_factor = process_weight(options['x_sec'], options['n_eff'])
            print(f"Luminosity and Cross Section weighting: {process_factor}")
            if 'Filtered' in sample: # apply filter efficiency if requires
                df['weight'] *= options['filter_eff']
                print(f"Filter efficiency applied: {options['filter_eff']}")
            if 'ProdAndDecay' in sample: # LHE reweight CP samples
                df = cp_weight(df) # LHE reweighting
        elif options['data']:
            opposite_sign = False # same_sign for qcd (data-driven)
            process_factor = 1 # take extrapolation factor to be 1 for now
            print(f"Extrapolation factor weighting: {process_factor}")
        # Apply weights
        df['weight'] *= process_factor # update central weight

        # Apply Selections
        n_before_cuts = len(df)
        df = cuts(df, opposite_sign, cfg['Selection'])
        n_after_cuts = len(df)
        print(f"Number of events initially: {n_before_cuts}")
        print(f"Number of events after selections: {n_after_cuts}: ({(n_after_cuts / n_before_cuts) * 100:.2f}% kept)")

        # Set class labels
        df["class_label"] = options['label']
        print(f"Assigned label {options['label']}")
        df["NN_weight"] = df["weight"] # initialise column that will store weight for NN

        # Select features
        df = df[cfg['Features']]

        # Save processed dataframe
        out_path = os.path.join(cfg['Paths']['proc_output'], sample)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        df.to_parquet(os.path.join(out_path, "merged_filtered.parquet"), engine="pyarrow")
        print("\n ------------------------------------------------- \n")



def main():
    cfg = yaml.safe_load(open("../config/config.yaml"))
    process_samples(cfg)

if __name__ == "__main__":
    main()