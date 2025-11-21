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



def create_even_dataset(path, df, train_frac=0.7):
    # -> save the files to train the EVEN Model (WARNING: these have ODD event numbers)
    logger.debug('\n Creating datasets for EVEN model training and validation:')

    # only events with an ODD event number should only be used to train and tune the EVEN model
    df = df[df['event'] % 2 == 1]

    # Find number of events to use for training
    n_train = int(len(df)*train_frac)
    df_train = df[:n_train] # training dataframe
    df_train.to_parquet(os.path.join(path, 'ShuffleMerge_EVENmodel_TRAIN.parquet'))
    logger.info(f"Saved {len(df_train)} events to 'ShuffleMerge_EVENmodel_TRAIN.parquet'")

    # Save training dataframe
    val_df = df[n_train:] # validation dataframe
    val_df.to_parquet(os.path.join(path, 'ShuffleMerge_EVENmodel_VAL.parquet'))
    logger.info(f"Saved {len(val_df)} events to 'ShuffleMerge_EVENmodel_VAL.parquet")
    print('-'*140)

    return True

def create_odd_dataset(path, df, train_frac=0.7):
    # -> save the files to train the ODD Model (WARNING: these have EVEN event numbers)
    logger.debug('\n Creating datasets for ODD model training and validation:')

    # only events with an EVEN event number should only be used to train and tune the ODD model
    df = df[df['event'] % 2 == 0]

    # Find number of events to use for training
    n_train = int(len(df)*train_frac)
    df_train = df[:n_train] # training dataframe
    df_train.to_parquet(os.path.join(path, 'ShuffleMerge_ODDmodel_TRAIN.parquet'))
    logger.info(f"Saved {len(df_train)} events to 'ShuffleMerge_ODDmodel_TRAIN.parquet'")

    # Save training dataframe
    val_df = df[n_train:] # validation dataframe
    val_df.to_parquet(os.path.join(path, 'ShuffleMerge_ODDmodel_VAL.parquet'))
    logger.info(f"Saved {len(val_df)} events to 'ShuffleMerge_ODDmodel_VAL.parquet")
    print('-'*140)

    return True


# Shuffle and merge files that have been pre-processed

def shuffle_merge(cfg, save_shards=False):
    # Merge all files into one dataframe
    merged_df = pd.DataFrame()
    for era in cfg['Setup']['eras']:
        logger.info(f"Loading era: {era}")
        era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
        # Load list of processed samples
        proc_list = os.path.join(cfg['Setup']['proc_output'], era, cfg['Setup']['channel'], f"processed_datasets.yaml")
        datasets = yaml.safe_load(open(proc_list))
        for file_path in datasets:
            logger.debug(f"Adding [{file_path.split('/')[-2]+file_path.split('/')[-1]}] from {era}")
            df = pd.read_parquet(file_path)
            df_dup = df[df.duplicated()]
            if len(df_dup) > 0:
                logger.warning(f"Found {len(df_dup)} duplicated events in {file_path.split('/')[-2]} - REMOVING")
                df = df.drop_duplicates(keep='first')
            merged_df = pd.concat([merged_df, df])
        print("="*140)
    # Shuffle the dataframe
    merged_df = merged_df.sample(frac=1, random_state=1879).reset_index(drop=True) # shuffle df
    # Apply normalisation for class balancing
    merged_df = normalise(merged_df)
    # Save total dataframe
    output_path = os.path.join(cfg['Setup']['output'], args.channel)
    logger.info(f"Output path is: {output_path}")
    logger.info(f"Total number of events is: {len(merged_df)}")
    print('-'*140)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    merged_df.to_parquet(os.path.join(output_path, "ShuffleMerge_ALL.parquet"), engine = "pyarrow")
    # Save dataframes for EVEN model:
    create_even_dataset(output_path, merged_df)
    # Save dataframes for ODD model:
    create_odd_dataset(output_path, merged_df)


def normalise(merged_df):
    # Initialise class_weight
    merged_df['class_weight'] = merged_df['weight']
    # Target sum of weights to be N events
    w_sum_target = len(merged_df)/2
    logger.debug(f"\nTotal number of events is {len(merged_df)} so targetting sum of class weights to be {w_sum_target:.1f} for each category\n")
    # Use only positive weights for class normalisation
    # Sum existing physics weights accross each category
    w_sum_taus = merged_df.loc[(merged_df['class_label'] == 0) & (merged_df['weight'] > 0), 'weight'].sum()
    w_sum_signal = merged_df.loc[(merged_df['class_label'] == 1) & (merged_df['weight'] > 0), 'weight'].sum()
    w_sum_cat = [w_sum_taus, w_sum_signal]
    logger.debug(f"Sum of original weights for Genuine Taus [label 0]: {w_sum_taus:.2f}")
    logger.debug(f"Sum of original weights for W [label 1]: {w_sum_signal:.2f}")
    # Calculate normalisation weights
    w_cat = [w_sum_target/w for w in w_sum_cat]
    logger.debug(f"Sum of original weights for Genuine Taus [label 0]: {w_sum_cat[0]:.2f} -> assigned category weight {w_cat[0]}")
    logger.debug(f"Sum of original weights for W [label 1]: {w_sum_cat[1]:.2f} -> assigned category weight {w_cat[1]}")
    # Apply appropriate NN weight
    merged_df.loc[merged_df['class_label'] == 0, 'class_weight'] *= w_cat[0]
    merged_df.loc[merged_df['class_label'] == 1, 'class_weight'] *= w_cat[1]
    return merged_df


def main():
    cfg = yaml.safe_load(open(f"../config/config_{args.channel}_FFs.yaml"))
    shuffle_merge(cfg)

if __name__ == "__main__":
    main()

