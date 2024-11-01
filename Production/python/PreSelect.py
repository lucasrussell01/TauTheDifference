import numpy as np
import pandas as pd
from glob import glob
import os
import json
import yaml
from utils import get_logger
from selection import Selector
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Plot histogram for a variable of choice")
    parser.add_argument('--channel', type=str, help="Channel to process", required=True)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()


args = get_args()
logger = get_logger(debug=args.debug)

# Preselection of HiggsDNA ouputs for classifier training

def save_skims(df, cfg, era, sample, gen_match='inc', logger=logger):
    # Drop unwanted features
    df = df[cfg['Features']]
    logger.debug(f"Dropping all features not specified in config")
    # Save the dataframe
    channel = cfg["Setup"]["channel"]
    out_path = os.path.join(cfg['Setup']['skim_output'], era, channel, sample)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_path = os.path.join(out_path, f"merged_skimmed_GEN{gen_match}.parquet")
    # Save the dataframe
    df.to_parquet(file_path, engine="pyarrow")
    logger.info(f"{len(df)} events saved to {file_path.split(channel+'/')[1]}")
    return file_path

def preselect_samples(cfg, era):
    # Preprocessing for signal background samples (skimming step)
    logger.info(f'Beginning preselection for era {era}')
    # Load configuration for the era, process and channel
    channel = cfg["Setup"]["channel"]
    era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
    channel_cfg = era_cfg[f'Channel_{channel}'] # Processes and Gen matching
    process_cfg = era_cfg['Process'] # For each Process: Datasets, N_eff, x_sec etc
    selector = Selector(logger)
    # Iterate over processes for the channel
    for process, process_options in channel_cfg.items():
        logger.debug(f"Process {process} was requested")
        # Iterate over datasets for the process
        for dataset, dataset_info in process_cfg[process].items():
            print('-'*140)
            logger.info(f"Processing {dataset}")
            # Load the dataset
            dataset_file = os.path.join(cfg['Setup']['input'], era,
                                channel, dataset, 'nominal/merged.parquet')
            df = pd.read_parquet(dataset_file)
            # Apply general selections and trigger matching
            # MuTau Channel Selections
            if channel == 'mt':
                df = selector.select_id_mt(df, cfg['Selection'])
                if process == 'Muon_DATA': # Data-driven QCD
                    df = selector.select_os(df, False)
                else:
                    df = selector.select_os(df, True)
                df = selector.mutau_trigger_match(df, cfg['Selection']['triggers'])
            # TauTau Channel Selections
            elif channel == 'tt':
                df = selector.select_id_tt(df, cfg['Selection'])
                if process == 'Tau_DATA': # Data-driven QCD
                    df = selector.select_os(df, False)
                else:
                    df = selector.select_os(df, True)
                df = selector.ditau_trigger_match(df, cfg['Selection']['triggers'])
            # Reject negative LHE CP weights
            if 'ProdAndDecay' in dataset:
                df = selector.cp_weight(df)
            # We may want to select one or more gen particles from a dataset
            if process_options['gen_match']:
                for gen_match in process_options['gen_match']:
                    if gen_match == 'tau':
                        df_tau = selector.select_gen_tau(df)
                        save_skims(df_tau, cfg, era, dataset, gen_match=gen_match)
                    elif gen_match == 'lep':
                        df_muon = selector.select_gen_lepton(df)
                        save_skims(df_muon, cfg, era, dataset, gen_match=gen_match)
                    elif gen_match == 'jet':
                        df_jet = selector.select_gen_jet(df)
                        save_skims(df_jet, cfg, era, dataset, gen_match=gen_match)
            else:
                logger.debug(f"No gen matching requested for {dataset}")
                save_skims(df, cfg, era, dataset, gen_match="inc") # inclusive
        print('='*140)


def main():
    # Load configuration for the desired channel
    cfg = yaml.safe_load(open(f"../config/config_{args.channel}.yaml"))
    for era in cfg['Setup']['eras']:
        preselect_samples(cfg, era)


if __name__ == "__main__":
    main()