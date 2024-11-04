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


# Reweighting and labelling of skimmed ouputs for classifier training


def save_skims(df, cfg, era, sample, gen_match='inc', logger=logger):
    # Save the dataframe
    channel = cfg["Setup"]["channel"]
    out_path = os.path.join(cfg['Setup']['proc_output'], era, channel, sample)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_path = os.path.join(out_path, f"merged_skimmed_GEN{gen_match}.parquet")
    # Save the dataframe
    df.to_parquet(file_path, engine="pyarrow")
    logger.info(f"{len(df)} events saved to {file_path.split(channel+'/')[1]}")
    return file_path


def apply_filter(df, filter):
    df['weight'] *= filter
    logger.debug(f"Applied filter: {filter}")
    return df


def label_df(df, class_label, proc_id, era, gen_match='inc'):
    df["class_label"] = class_label
    df['process_id'] = proc_id
    logger.info(f"Assigned label {class_label} and Process ID: {proc_id} - [matched {gen_match}]")
    if era == 'Run3_2022':
        df['era'] = 1
    elif era == 'Run3_2022EE':
        df['era'] = 2
    elif era == 'Run3_2023':
        df['era'] = 3
    elif era == 'Run3_2023BPix':
        df['era'] = 4
    else:
        df['era'] = -1
    logger.debug(f"Assigned era label for {era}")
    return df


def reweight_mc(df, xsec, n_eff, lumi):
    # Luminoisty and cross section weighting
    process_factor = (xsec * lumi) / n_eff
    df['weight'] *= process_factor
    logger.debug(f"Reweighting to xsec: {xsec}, N_eff: {n_eff}, Luminosity: {lumi}")
    return df


def process_samples(cfg, era):
    # List of samples that have been processed (used for ShuffleMerge)
    processed_datasets = []
    # Preprocessing for signal background samples (skimming step)
    logger.info(f'Beginning processing for era {era}')
    print('*'*140)
    # Load configuration for the era, process and channel
    channel = cfg["Setup"]["channel"]
    era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
    channel_cfg = era_cfg[f'Channel_{channel}'] # Processes and Gen matching
    process_cfg = era_cfg['Process'] # For each Process: Datasets, N_eff, x_sec etc
    # Iterate over processes for the channel
    for process, process_options in channel_cfg.items():
        logger.info(f"Loading skimmed datasets for {process}")
        # Iterate over datasets for the proces
        for dataset, dataset_info in process_cfg[process].items():
            print('-'*140)
            logger.info(f"Processing {dataset}")
            # GEN MATCHED SAMPLES
            if process_options['gen_match']:
                logger.info(f"Gen matching options found for {process}")
                for index_gen, gen_match in enumerate(process_options['gen_match']): # use index to find correct label
                    logger.debug(f"Gen matching: {gen_match}")
                    # Load the gen matched dataset
                    dataset_file = os.path.join(cfg['Setup']['skim_output'], era,
                                    channel, dataset, f'merged_skimmed_GEN{gen_match}.parquet')
                    logger.debug(f"Loading {dataset_file.split('/')[-1]}")
                    df = pd.read_parquet(dataset_file)
                    # Add labels (class, process, era)
                    df = label_df(df, process_options['label'][index_gen], process_options['proc_id'][index_gen], era, gen_match)
                    # Reweight the dataset (can only have MC here anyway since gen matched)
                    df = reweight_mc(df, dataset_info['x_sec'], dataset_info['n_eff'], era_cfg['Params']['Luminosity'])
                    # Save the dataframe
                    processed_datasets.append(save_skims(df, cfg, era, dataset, gen_match=gen_match))
            # INCLUSIVE GEN SAMPLES
            else:
                logger.debug(f"No gen matching requested for {dataset}")
                logger.debug(f"Label {process_options['label']} and Process ID: {process_options['proc_id']}")
                # Load the inclusive datset
                dataset_file = os.path.join(cfg['Setup']['skim_output'], era,
                                channel, dataset, f'merged_skimmed_GENinc.parquet')
                logger.debug(f"Loading {dataset_file.split('/')[-1]}")
                df = pd.read_parquet(dataset_file)
                # Add labels (class, process, era)
                df = label_df(df, process_options['label'], process_options['proc_id'], era)
                if 'DATA' not in process: # Monte Carlo
                    # Reweight the dataset
                    df = reweight_mc(df, dataset_info['x_sec'], dataset_info['n_eff'], era_cfg['Params']['Luminosity'])
                    # Apply filter efficiency
                    if 'Filtered' in dataset:
                        df = apply_filter(df, dataset_info['filter_eff'])
                else: # Same sign QCD estimate
                    logger.warning('Extrapolation factor needed')
                # Save the dataframe
                processed_datasets.append(save_skims(df, cfg, era, dataset, gen_match="inc"))
        print('='*140)
    # Return the list of processed datasets
    return processed_datasets


def main():
    args = get_args()
    # Load configuration for the desired channel
    cfg = yaml.safe_load(open(f"../config/config_{args.channel}.yaml"))
    for era in cfg['Setup']['eras']:
        processed_ds = process_samples(cfg, era)
        # Save the list of processed datasets (used for shuffle merge)
        output_file = os.path.join(cfg['Setup']['proc_output'], era, args.channel, f"processed_datasets.yaml")
        yaml.dump(processed_ds, open(output_file, 'w'))
        logger.info(f"Processed datasets written to {output_file}")


if __name__ == "__main__":
    main()