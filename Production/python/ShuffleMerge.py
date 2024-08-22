import os
import pandas as pd
import numpy as np
import yaml
from Split import train_eval_split_shards, split_data

# Shuffle and merge files that have been pre-processed

def shuffle_merge(cfg, save_shards=False):
    # Merge all files into one dataframe
    merged_df = pd.DataFrame()
    for sample in cfg['Datasets'].keys():
        file_path = os.path.join(cfg['Paths']['proc_output'], sample, 'merged_filtered.parquet')
        df = pd.read_parquet(file_path)
        merged_df = pd.concat([merged_df, df])
    merged_df = merged_df.sample(frac=1).reset_index(drop=True) # shuffle df
    # Apply normalisation for class balancing
    merged_df = normalise(merged_df)
    # Save merged dataframe
    merged_df.to_parquet(os.path.join(cfg['Paths']['merge_output'], "ShuffleMerge_ALL.parquet"), engine = "pyarrow")
    split_data(cfg['Paths']['merge_output'], 0.60, 0.15, 0.25) # split into train, val, eval
    if save_shards:
        save_shards_df(merged_df, cfg['Paths']['merge_output'])
        train_eval_split_shards(os.path.join(cfg['Paths']['merge_output'], 'shards'), 0.7)

def normalise(merged_df):
    # Target sum of weights to be N events
    w_sum_target = len(merged_df)/3
    print(f"\nTotal number of events is {len(merged_df)} so targetting sum of NN weights to be {w_sum_target:.1f} for each category\n")

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
    merged_df.loc[merged_df['class_label'] == 0, 'NN_weight'] *= w_cat[0]
    merged_df.loc[merged_df['class_label'].isin([11, 12]), 'NN_weight'] *= w_cat[1]
    merged_df.loc[merged_df['class_label'] == 2, 'NN_weight'] *= w_cat[2]

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

