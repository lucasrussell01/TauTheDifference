import os
import shutil
import glob
import random
import pandas as pd

def train_eval_split_shards(input_dir, train_fraction):
    eval_fraction = 1-train_fraction
    train_dir = os.path.join(input_dir, '../train')
    eval_dir = os.path.join(input_dir, '../evaluation')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    file_list = glob.glob(os.path.join(input_dir, "*.parquet"))
    random.shuffle(file_list)
    file_list = [file.split('/')[-1] for file in file_list]
    # split files
    num_files = len(file_list)
    num_train_files = int(num_files * train_fraction)
    num_eval_files = int(num_files * eval_fraction)
    print(f"{num_train_files} for training, {num_eval_files} for evaluation.")
    # move files to relevant directory
    for file_name in file_list[:num_train_files]:
        src_file = os.path.join(input_dir, file_name)
        dst_file = os.path.join(train_dir, file_name)
        shutil.copy(src_file, dst_file)
    for file_name in file_list[num_train_files:num_train_files+num_eval_files]:
        src_file = os.path.join(input_dir, file_name)
        dst_file = os.path.join(eval_dir, file_name)
        shutil.copy(src_file, dst_file)


def split_data(path, train_fraction, val_fraction, eval_fraction):
    print(f"Splitting data into fractions Training: {train_fraction}, Validation: {val_fraction}, Evaluation: {eval_fraction}")
    df = pd.read_parquet(os.path.join(path, 'ShuffleMerge_ALL.parquet'))
    train_size = int(len(df) * train_fraction)
    val_size = int(len(df) * val_fraction)
    print(f"Event Statistics - Training: {train_size}, Validation: {val_size}, Evaluation: {len(df)-train_size-val_size}")
    train_df = df[:train_size]
    val_df = df[train_size:train_size+val_size]
    eval_df = df[train_size+val_size:]
    train_df.to_parquet(os.path.join(path, 'ShuffleMerge_TRAIN.parquet'))
    val_df.to_parquet(os.path.join(path, 'ShuffleMerge_VAL.parquet'))
    eval_df.to_parquet(os.path.join(path, 'ShuffleMerge_EVAL.parquet'))
    return train_df, val_df, eval_df

# Separate the shards:
input_dir = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/shards/'
train_eval_split_shards(input_dir, 0.7)

# Split the main df
input_dir = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/'
split_data(input_dir, 0.60, 0.15, 0.25)