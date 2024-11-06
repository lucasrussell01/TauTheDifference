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
    parser = argparse.ArgumentParser(description="Select events of interest from HiggsDNA outputs for classifier training")
    parser.add_argument('--channel', type=str, help="Channel to process", required=True)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--extrapolate', action='store_true', help="Extrapolate QCD")
    return parser.parse_args()


args = get_args()
logger = get_logger(debug=args.debug)

# Preselection of HiggsDNA ouputs for classifier training

def save_skims(df, cfg, era, sample, gen_match='inc', extrapolate=False, logger=logger):
    # Drop unwanted features
    df = df[cfg['Features']]
    logger.debug(f"Dropping all features not specified in config")
    # Save the dataframe
    channel = cfg["Setup"]["channel"]
    if extrapolate:
        out_path = os.path.join(cfg['Setup']['skim_output'], "ExtrapolateQCD", era, channel, sample)
        file_path = os.path.join(out_path, f"merged_skimmed_GEN{gen_match}_SAMESIGN.parquet")
    else:
        out_path = os.path.join(cfg['Setup']['skim_output'], era, channel, sample)
        file_path = os.path.join(out_path, f"merged_skimmed_GEN{gen_match}.parquet")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # Save the dataframe
    df.to_parquet(file_path, engine="pyarrow")
    logger.info(f"{len(df)} events saved to {file_path.split(channel+'/')[1]}")
    return file_path

def preselect_samples(cfg, era, extrapolateQCD=False):
    # Preprocessing for signal background samples (skimming step)
    print('\n')
    print('*'*140)
    if extrapolateQCD:
        logger.warning(f'Beginning preselection for era {era} for QCD extrapolation')
    else:
        logger.info(f'Beginning preselection for era {era}')
    print('*'*140, '\n')
    # Load configuration for the era, process and channel
    channel = cfg["Setup"]["channel"]
    era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
    channel_cfg = era_cfg[f'Channel_{channel}'] # Processes and Gen matching
    process_cfg = era_cfg['Process'] # For each Process: Datasets, N_eff, x_sec etc
    selector = Selector(logger)
    # Iterate over processes for the channel
    for process, process_options in channel_cfg.items():
        logger.info(f"Process {process} was requested")
        # Iterate over datasets for the process
        for dataset, dataset_info in process_cfg[process].items():
            print('-'*140)
            logger.info(f"Processing {dataset}")
            # Load the dataset
            dataset_file = os.path.join(cfg['Setup']['input'], era,
                                channel, dataset, 'nominal/merged.parquet')
            df = pd.read_parquet(dataset_file)
            # Apply general selections and trigger matching
            df = selector.check_sign_weights(df) # drop negative weights (affect training)
            # MuTau Channel Selections
            if channel == 'mt':
                df = selector.select_id_mt(df, cfg['Selection'])
                df = selector.mutau_trigger_match(df, cfg['Selection']['triggers'])
            # ETau Channel Selections
            elif channel == 'et':
                logger.error("ETau channel not implemented")
                raise NotImplementedError
            # TauTau Channel Selections
            elif channel == 'tt':
                df = selector.select_id_tt(df, cfg['Selection'])
                df = selector.ditau_trigger_match(df, cfg['Selection']['triggers'])
            # Pair Sign Selection
            if ((process == 'Muon_DATA' and channel == 'mt') or (process == 'Tau_DATA' and channel == 'tt')
                or extrapolateQCD): # Data-driven or QCD estimation
                logger.warning('Selecting same sign pairs')
                df = selector.select_os(df, False)
            else:
                df = selector.select_os(df, True)
            # Reject negative LHE CP weights
            if 'ProdAndDecay' in dataset:
                df = selector.cp_weight(df)
            # We may want to select one or more gen particles from a dataset
            if process_options['gen_match']:
                for gen_match in process_options['gen_match']:
                    if gen_match == 'tau':
                        df_tau = selector.select_gen_tau(df)
                        save_skims(df_tau, cfg, era, dataset, gen_match=gen_match, extrapolate=extrapolateQCD)
                    elif gen_match == 'lep':
                        df_muon = selector.select_gen_lepton(df)
                        save_skims(df_muon, cfg, era, dataset, gen_match=gen_match, extrapolate=extrapolateQCD)
                    elif gen_match == 'jet':
                        df_jet = selector.select_gen_jet(df)
                        save_skims(df_jet, cfg, era, dataset, gen_match=gen_match, extrapolate=extrapolateQCD)
            else:
                logger.debug(f"No gen matching requested for {dataset}")
                save_skims(df, cfg, era, dataset, gen_match="inc", extrapolate=extrapolateQCD) # inclusive
        print('\n')
        print('='*140)


def main():
    # Load configuration for the desired channel
    cfg = yaml.safe_load(open(f"../config/config_{args.channel}.yaml"))
    for era in cfg['Setup']['eras']:
        preselect_samples(cfg, era, args.extrapolate)


if __name__ == "__main__":
    main()