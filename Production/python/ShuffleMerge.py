import os
import pandas as pd
import numpy as np
import yaml
from Split import train_eval_split_shards, split_data



def create_even_dataset(path, df, train_frac=0.7):
    # -> save the files to train the EVEN Model (WARNING: these have ODD event numbers)
    print('\n Creating datasets for EVEN model training and validation:')

    # only events with an ODD event number should only be used to train and tune the EVEN model
    df = df[df['event'] % 2 == 1]

    # Find number of events to use for training
    n_train = int(len(df)*train_frac)
    df_train = df[:n_train] # training dataframe
    df_train.to_parquet(os.path.join(path, 'ShuffleMerge_EVENmodel_TRAIN.parquet'))
    print(f"Saved {len(df_train)} events to 'ShuffleMerge_EVENmodel_TRAIN.parquet'")

    # Save training datafram
    val_df = df[n_train:] # validation dataframe
    val_df.to_parquet(os.path.join(path, 'ShuffleMerge_EVENmodel_VAL.parquet'))
    print(f"Saved {len(val_df)} events to 'ShuffleMerge_EVENmodel_VAL.parquet")
    print('---------------------------------------------------------------')

    return True

def create_odd_dataset(path, df, train_frac=0.7):
    # -> save the files to train the ODD Model (WARNING: these have EVEN event numbers)
    print('\n Creating datasets for ODD model training and validation:')

    # only events with an EVEN event number should only be used to train and tune the ODD model
    df = df[df['event'] % 2 == 0]

    # Find number of events to use for training
    n_train = int(len(df)*train_frac)
    df_train = df[:n_train] # training dataframe
    df_train.to_parquet(os.path.join(path, 'ShuffleMerge_ODDmodel_TRAIN.parquet'))
    print(f"Saved {len(df_train)} events to 'ShuffleMerge_ODDmodel_TRAIN.parquet'")

    # Save training datafram
    val_df = df[n_train:] # validation dataframe
    val_df.to_parquet(os.path.join(path, 'ShuffleMerge_ODDmodel_VAL.parquet'))
    print(f"Saved {len(val_df)} events to 'ShuffleMerge_ODDmodel_VAL.parquet")
    print('---------------------------------------------------------------')

    return True


# Shuffle and merge files that have been pre-processed

def shuffle_merge(cfg, save_shards=False):
    # Merge all files into one dataframe
    merged_df = pd.DataFrame()
    for era in cfg['Setup']['eras']:
        era_cfg = yaml.safe_load(open(f"../config/{era}.yaml"))
        datasets = era_cfg[f'samples_{cfg["Setup"]["channel"]}']
        for sample in datasets.keys():
            file_path = os.path.join(cfg['Setup']['skim_output'], sample, era, 'merged_filtered.parquet')
            df = pd.read_parquet(file_path)
            merged_df = pd.concat([merged_df, df])
    merged_df = merged_df.sample(frac=1, random_state=1879).reset_index(drop=True) # shuffle df
    # Apply normalisation for class balancing
    merged_df = normalise(merged_df)
    # Save total dataframe
    if not os.path.exists(cfg['Setup']['output']):
        os.makedirs(cfg['Setup']['output'])
    merged_df.to_parquet(os.path.join(cfg['Setup']['output'], "ShuffleMerge_ALL.parquet"), engine = "pyarrow")
    # Save dataframes for EVEN model:
    create_even_dataset(cfg['Setup']['output'], merged_df)
    # Save dataframes for ODD model:
    create_odd_dataset(cfg['Setup']['output'], merged_df)
    # split_data(os.path.join(cfg['Setup']['output'], p), 0.50, 0.25, 0.25) # split into train, val, eval
    # if save_shards:
    #     save_shards_df(df_split, cfg['Setup']['output'])
    #     train_eval_split_shards(os.path.join(cfg['Setup']['output'], p, 'shards'), 0.7)

def normalise(merged_df):
    # Target sum of weights to be N events
    w_sum_target = len(merged_df)/3
    print(f"\nTotal number of events is {len(merged_df)} so targetting sum of class weights to be {w_sum_target:.1f} for each category\n")

    # Sum existing physics weights accross each category
    print("Category statistics:")
    w_sum_taus = merged_df.loc[merged_df['class_label'] == 0, 'weight'].sum()
    w_sum_signal = merged_df.loc[merged_df['class_label'].isin([11, 12]), 'weight'].sum()
    w_sum_bkg = merged_df.loc[merged_df['class_label'] == 2, 'weight'].sum()
    w_sum_cat = [w_sum_taus, w_sum_signal, w_sum_bkg]
    # Calculate normalisation weights
    w_cat = [w_sum_target/w for w in w_sum_cat]
    print(f"Sum of original weights for Genuine Taus [label 0]: {w_sum_cat[0]:.2f} -> assigned category weight {w_cat[0]}")
    print(f"Sum of original weights for Signal [label 1 (11 and 12)]: {w_sum_cat[1]:.2f} -> assigned category weight {w_cat[1]}")
    print(f"Sum of original weights for Background  [label 2]: {w_sum_cat[2]:.2f} -> assigned category weight {w_cat[2]}\n")

    # Apply appropriate NN weight
    merged_df.loc[merged_df['class_label'] == 0, 'class_weight'] *= w_cat[0]
    merged_df.loc[merged_df['class_label'].isin([11, 12]), 'class_weight'] *= w_cat[1]
    merged_df.loc[merged_df['class_label'] == 2, 'class_weight'] *= w_cat[2]

    return merged_df

def save_shards_df(merged_df, out_dir, shard_size=50000):
    # Save dataframe into several shards
    num_shards = len(merged_df) // shard_size
    out_shard = os.path.join(out_dir, 'shards')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_shard):
        os.makedirs(out_shard)
    for i in range(num_shards):
        shard_df = merged_df[i*shard_size:(i+1)*shard_size]
        shard_file_path = os.path.join(out_shard, f'ShuffleMerge_{i}.parquet')
        shard_df.to_parquet(shard_file_path)
        print(f"Saved shard {i} with [{shard_size}] Events")
    if len(merged_df)%shard_size != 0: # if any remnants
        remaining_df = merged_df[num_shards*shard_size:]
        remaining_file_path = os.path.join(out_shard, f'ShuffleMerge_{num_shards}_remnant.parquet')
        remaining_df.to_parquet(remaining_file_path)
        print(f"Saved shard remnant {num_shards} with [{len(merged_df)%shard_size}] Events\n")



def main():
    cfg = yaml.safe_load(open("../config/config.yaml"))
    shuffle_merge(cfg)

if __name__ == "__main__":
    main()

