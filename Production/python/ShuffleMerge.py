import os
import pandas as pd
import numpy as np

# Shuffle and merge files that have been processed by SkimFiles.py

base_dir = '/vols/cms/lcr119/offline/HiggsCP/data/processed/2022/tt'

samples = ['DYto2L_M-50_madgraphMLM', 'DYto2L_M-50_madgraphMLM_ext1', 'DYto2L_M-50_1J_madgraphMLM',
           'DYto2L_M-50_2J_madgraphMLM', 'DYto2L_M-50_3J_madgraphMLM', 'DYto2L_M-50_4J_madgraphMLM',
           'Tau_Run2022C', 'Tau_Run2022D', 'GluGluHTo2Tau_UncorrelatedDecay_SM_Filtered_ProdAndDecay',
           'GluGluHTo2Tau_UncorrelatedDecay_MM_Filtered_ProdAndDecay', 'GluGluHTo2Tau_UncorrelatedDecay_CPodd_Filtered_ProdAndDecay',
           'VBFHToTauTau_UncorrelatedDecay_Filtered']
            # 'GluGluHToTauTau_M125', 'VBFHToTauTau_M125',

out_dir = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt'

# Merge all files into one dataframe
merged_df = pd.DataFrame()
for sample in samples:
    file_path = os.path.join(base_dir, sample, 'merged_filtered.parquet')
    df = pd.read_parquet(file_path)
    merged_df = pd.concat([merged_df, df])
merged_df = merged_df.sample(frac=1).reset_index(drop=True) # shuffle df

# Print statistics
# label_counts = merged_df['class_label'].value_counts()
# print(f"\nSummary of dataframe statistics [Total: {merged_df.shape[0]} Events]:")
# for label, count in label_counts.items():
#     w_sum = merged_df.loc[merged_df['class_label'] == label, 'weight'].sum()
#     print(f"Label {label}: {count} events, Sum of weights: {w_sum:.2f}")
    
w_sum_target = len(merged_df)/3 
print(f"\nTotal number of events is {len(merged_df)} so targetting sum of NN weights to be {w_sum_target:.1f} for each category\n")
merged_df['NN_weight'] = merged_df["weight"] # initialise column that will store weight for NN


# Category weight calculations
print("Category statistics:")
w_sum_taus = merged_df.loc[merged_df['class_label'] == 0, 'weight'].sum()
w_sum_signal = merged_df.loc[merged_df['class_label'].isin([11, 12]), 'weight'].sum()
w_sum_bkg = merged_df.loc[merged_df['class_label'] == 2, 'weight'].sum()
w_sum_cat = [w_sum_taus, w_sum_signal, w_sum_bkg]

# Calculate appropriate category normalisation
# w_cat = [np.sum(w_sum_cat)/w for w in w_sum_cat]
w_cat = [w_sum_target/w for w in w_sum_cat]

print(f"Sum of original weights for Genuine Taus [label 0]: {w_sum_cat[0]:.2f} -> assigned category weight {w_cat[0]}")
print(f"Sum of original weights for Signal [label 1 (11 and 12)]: {w_sum_cat[1]:.2f} -> assigned category weight {w_cat[1]}")
print(f"Sum of original weights for Background  [label 2]: {w_sum_cat[2]:.2f} -> assigned category weight {w_cat[2]}\n")


# Make appropriate NN weight
merged_df.loc[merged_df['class_label'] == 0, 'NN_weight'] *= w_cat[0]
merged_df.loc[merged_df['class_label'].isin([11, 12]), 'NN_weight'] *= w_cat[1]
merged_df.loc[merged_df['class_label'] == 2, 'NN_weight'] *= w_cat[2]


# # Check new weights
# w_sum_taus = merged_df.loc[merged_df['class_label'] == 0, 'NN_weight'].sum()
# w_sum_signal = merged_df.loc[merged_df['class_label'].isin([11, 12]), 'NN_weight'].sum()
# w_sum_bkg = merged_df.loc[merged_df['class_label'] == 2, 'NN_weight'].sum()
# w_sum_cat = [w_sum_taus, w_sum_signal, w_sum_bkg]
# print("*** Summary of final statistics ***")
# print(f"Label 0 (Genuine Taus): {w_sum_cat[0]:.2f}")
# print(f"Label 1 (11/12) (Signal):  {w_sum_cat[1]:.2f}")
# print(f"Label 2 (Background): {w_sum_cat[2]:.2f}\n")

# Save merged dataframe
out_shard = os.path.join(out_dir, 'shards')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
if not os.path.exists(out_shard):
    os.makedirs(out_shard)
merged_df.to_parquet(os.path.join(out_dir, "ShuffleMerge_ALL.parquet"), engine = "pyarrow")

# Also save in several shards
shard_size = 50000
num_shards = len(merged_df) // shard_size
# Complete files:
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

    
print(f"Total of {len(merged_df)} events saved to: {out_dir}")

