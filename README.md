# Tau The Difference

Separate Higgs to Tau Tau decays from Genuine and Fake Backgrounds.

Channels currently supported:
- Fully hadronic ($\tau_h\tau_h$)

Models currently supported:
- XGBClassifier (optimal so far)
- Simple DNN

## Production

The production of training tuples starts with `parquet` files that are NanoAOD skims processed  by the [HiggsDNA](https://gitlab.cern.ch/dwinterb/HiggsDNA) framework for H $\to\tau\tau$ analyses.

The code necessary to producte samples for training (and validating/evaluating) models is in the `Production` directory.

### Configuration

In the `config` subdirectory, there are `yaml` files detailing the datasets to be used for each era, as well as relevant run parameters.

Currently available:
- `Run3_2022`
- `Run3_2022EE`
- `Run3_2023`
- `Run3_2023BPix`

The central production is controlled by `config.yaml`. Here the input/output paths, eras to be used, and selection to apply is specified.

### Execution

Production of the `ShuffleMerge` files used for training can be done in one command:
```
./scripts/run_prod.sh
```

This will execute `python/PreProcess.py` and `python/ShuffleMerge.py`.

The `PreProcess.py` script applies selections to the input datasets, adds class and era labels, and any relevant physics weihts (lumi*xs, filter efficiencies, LHE etc...)

The `ShuffleMerge.py` mixed all samples, and creates a `class_weight` where all three categories (genuine tau (0), higgs signal (1) and background (2)) are balanced. This ensures that the classes are considered equally during training.

### Output

Running these steps will produce several files:
- `ShuffleMerge_ALL.parquet` which contains all events (useful for analysis histograms etc)
- `ShuffleMerge_EVENmodel_Train` and `ShuffleMerge_EVENmodel_Val` to train and validate models that will be **applied** to EVEN events (these files contain ODD events only)
- `ShuffleMerge_ODDmodel_Train` and `ShuffleMerge_ODDmodel_Val` to train and validate models that will be **applied** to ODD events (these files contain EVEN events only)

These datasets can be found in the output directory specified in teh central `config.yaml` file.

## Analysis

Plotting and analytics for dataframes.

## Training

Model training and validation.

## Evaluation

Inference (applying training), plotting tools (confusion matrices, score distributions, ROC)
