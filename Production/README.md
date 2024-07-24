# Production

This directory contains the tools to tranasform `HiggsDNA` outputs into the shuffled and merged files to be used for model training.

## PreProcessing

PreProcessing is done using `PreProcess.py`.

The merged parquet files from `HiggsDNA` are used as inputs, opposite (or same for data-driven QCD) pairs are selected, relevant features are kept, and MC events are weighted by: `Luminosity*CrossSection/EffectiveEvents`. 

Labels are assigned to the classes:
- `0`: Genuine taus
- `1`: Higgs [`11` ggH and `12` VBF]
- `2`: Fake taus


## Shuffle and Merge

Input tuples are made from preprocessed inputs with `ShuffleMerge.py`

Different samples are shuffled and merged into one dataframe, a new column `NN_weight` is added which normalises the three classes to have equal weights, so that trained models assign equal importance.

`Split.py` then separates the files into Training, Validation and Testing datasets.