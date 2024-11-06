import os
import pandas as pd
import numpy as np
import yaml
from utils import get_logger
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Shuffle and Merge processed files")
    parser.add_argument('--channel', type=str, help="Channel to process", required=True)
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    return parser.parse_args()


args = get_args()
logger = get_logger(debug=args.debug)

def get_extrapolation_factor(n_data, n_mc, channel):
    logger.debug(f'Number of data events: {n_data}')
    logger.debug(f'Number of weighted MC events: {n_mc}')
    n_scale = n_data - n_mc # expected number of events to scale to
    logger.debug(f'Number of events to scale to: {n_scale}')
    if channel == 'mt':
        n_scale *= 1.12 # factor for muons
    factor = n_scale/n_data # fraction of events that should be kept (reweighted)
    logger.info(f"Calculated factor to estimate QCD for {channel}: {factor} ({n_data}->{n_scale:.1f} events)")
    return factor

def expected_events(cfg, era):
    # Load configuration for the era, process and channel
    channel = cfg["Setup"]["channel"]
    era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
    channel_cfg = era_cfg[f'Channel_{channel}'] # Processes and Gen matching
    process_cfg = era_cfg['Process'] # For each Process: Datasets, N_eff, x_sec etc
    # Event numbers
    data_events = 0 # tracks number of same sign data events
    mc_events = 0 # tracks number of same sign MC events
    # Processes of interest
    data_processes = ['Muon_DATA', 'Tau_DATA']
    mc_processes = ['DY', 'TTBar', 'ST', 'WJets', 'Diboson']
    # Iterate over processes for the channel
    for process, process_options in channel_cfg.items():
        logger.info(f"Process {process} was requested")
        if process in data_processes:
            data = True
            logger.info(f"Data process {process} identified")
        elif process in mc_processes:
            data = False
            logger.info(f"MC process {process} identified (to be substracted)")
        else:
            continue
        # Iterate over datasets for the process
        for dataset, dataset_info in process_cfg[process].items():
            print('-'*140)
            logger.info(f"Loading {dataset}")
            if data:
                dataset_file = os.path.join(cfg['Setup']['proc_output'], 'ExtrapolateQCD', era,
                                        channel, dataset, f'merged_skimmed_GENinc_SAMESIGN.parquet')
                df = pd.read_parquet(dataset_file)
                logger.debug(f"Sum of weights is {df['weight'].sum()}")
                data_events += df['weight'].sum()
            else: # gen matches to jets
                dataset_file = os.path.join(cfg['Setup']['proc_output'], 'ExtrapolateQCD', era,
                                        channel, dataset, f'merged_skimmed_GENjet_SAMESIGN.parquet')
                df = pd.read_parquet(dataset_file)
                logger.debug(f"Sum of weights is {df['weight'].sum()}")
                mc_events += df['weight'].sum()
        print('\n')
        print('='*140)
    # Calculate the expected number of events
    factor = get_extrapolation_factor(data_events, mc_events, channel)
    return factor


def main():
    era_factors = [] # store factors for the eras in order
    args = get_args()
    # Load configuration for the desired channel
    cfg = yaml.safe_load(open(f"../config/config_{args.channel}.yaml"))
    for era in cfg['Setup']['eras']:
        factor = expected_events(cfg, era)
        era_factors.append(factor)
    print('\n\n\n')
    logger.info(f"QCD estimation factor for eras: {cfg['Setup']['eras']}:")
    logger.info(era_factors)


if __name__ == "__main__":
    main()