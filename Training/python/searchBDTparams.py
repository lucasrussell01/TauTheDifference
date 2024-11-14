import xgboost as xgb
import optuna
from train_BDT import load_ds, AMS
import os
import yaml
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from statsmodels.stats.weightstats import DescrStatsW
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for XGBoost")
    parser.add_argument('--channel', type=str, help="Channel to optimise", required=True)
    parser.add_argument('--n_trials', type=int, help="Number of trials to attempt")
    parser.add_argument('--n_cores', type=int, help="Number of cores to use")
    parser.add_argument('--study_name', type=str, help="Name of study (can use to resume)")
    parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()

def validation(model, x, y, w_NN, w_phys, parity):
    # Get predictions
    y_pred_proba = model.predict_proba(x) # raw score
    y_pred = y_pred_proba.argmax(axis=1) # predicted label
    # Find events classified as Higgs
    y_pred_higgs = y_pred_proba[:, 1][y_pred==1]   # get raw Higgs scores of events classified as Higgs
    w_pred_higgs = w_phys[y_pred==1]          # get weights for those events
    y_higgs = y[y_pred==1]                # truth for Higgs classified events
    # Optimised binning (flat signal)
    n_bins = 5
    w_perc = DescrStatsW(y_pred_higgs[(y_higgs == 1)], weights=w_pred_higgs[(y_higgs == 1)]).quantile(np.linspace(0, 1, n_bins+1)[1:-1]) # percentiles
    bins = np.concatenate([[0.33], np.array(w_perc), [1]])
    # Histogram signal and background
    y_higgs = y[y_pred==1]                # truth for Higgs classified events
    s_counts = np.histogram(y_pred_higgs[(y_higgs == 1)], bins=bins, weights=w_pred_higgs[(y_higgs == 1)])[0]
    bkg_counts = np.histogram(y_pred_higgs[(y_higgs != 1)], bins=bins, weights=w_pred_higgs[(y_higgs != 1)])[0]
    # AMS Score
    ams_bybin = AMS(s_counts, bkg_counts)
    ams = np.sqrt(np.sum(ams_bybin**2))
    # print("AMS Score (bin by bin):", ams)
    print(f"AMS {parity}: {ams}")
    return ams


def train_model(x_train, y_train, w_train, parity, param):
    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**param)

    model.fit(x_train, y_train, sample_weight=w_train)

    return model


def objective(trial):

    # Optimise the sum of the two AMS scores

    param = {
        "verbosity": 0,
        'objective': 'multi:softmax',
        # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]), # booster: can add , "gblinear"
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000), # n estimators
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True), # L2 regularization weight.
        # "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True), # L1 regularization weight.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0), # sampling
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.7),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 10)
    }

    if args.gpu:
        # enable gpu training
        param['device'] = 'cuda'

        # Train even model
        model_even = train_model(x_train_gpu_EVEN, y_train_gpu_EVEN, w_train_gpu_EVEN, "EVEN", param)
        ams_even = validation(model_even, x_val_gpu_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_gpu_ODD, y_train_gpu_ODD, w_train_gpu_ODD, "ODD", param)
        ams_odd = validation(model_even, x_val_gpu_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")

    else:
        # Train even model
        model_even = train_model(x_train_EVEN, y_train_EVEN, w_train_EVEN, "EVEN", param)
        ams_even = validation(model_even, x_val_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN, "EVEN")

        # Train odd model
        model_odd = train_model(x_train_ODD, y_train_ODD, w_train_ODD, "ODD", param)
        ams_odd = validation(model_even, x_val_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD, "ODD")



    if abs(ams_even - ams_odd)/(ams_even + ams_odd) > 0.04: # allow a 4% difference in total AMS ~ 8% in between the two
        return 0 # effectvely veto this model
    else:
        return ams_even + ams_odd - abs(ams_even - ams_odd)


def main():


    print(f"Optimizing hyperparameters for XGBoost model with {args.n_trials} trials")

    if args.n_trials is not None:

        # Optuna study to optimize hyperparameters
        study = optuna.create_study(direction="maximize", study_name=args.study_name,
                                storage=f"sqlite:///hyperlogs/{args.study_name}.db?timeout=10000", load_if_exists=True)
        # Begin search
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_cores)

        # Summary
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))



if __name__ == "__main__":
    # Configuration of tuning via args
    args = get_args()
    if args.gpu:
        import cupy as cp
    if args.n_cores is None:
        args.n_cores = -1

    print("Loading config and datasets")
    # Find correct dataset to use and load config
    if args.channel == 'tt':
        cfg = yaml.safe_load(open("../config/tt/BDTHyperOpt_config.yaml"))
        if args.cut=='medium':
            data_path = 'input_path'
        elif args.cut=='tight':
            data_path = 'input_path_tight'
        elif args.cut=='vtight':
            data_path = 'input_path_vtight'
    elif args.channel == 'mt':
        cfg = yaml.safe_load(open("../config/mt/BDTHyperOpt_config.yaml"))
        data_path = 'input_path'
    elif args.channel == 'et':
        cfg = yaml.safe_load(open("../config/et/BDTHyperOpt_config.yaml"))
        data_path = 'input_path'

    # Load datasets
    # EVEN MODEL
    x_train_EVEN, y_train_EVEN, w_train_EVEN = load_ds(os.path.join(cfg['Setup'][data_path], f'ShuffleMerge_EVENmodel_TRAIN.parquet'),
                                                       cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'])
    x_val_EVEN, y_val_EVEN, w_NN_val_EVEN, w_phys_val_EVEN = load_ds(os.path.join(cfg['Setup'][data_path], 'ShuffleMerge_EVENmodel_VAL.parquet'),
                                                                    cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    # ODD MODEL
    x_train_ODD, y_train_ODD, w_train_ODD = load_ds(os.path.join(cfg['Setup'][data_path], f'ShuffleMerge_ODDmodel_TRAIN.parquet'),
                                                       cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'])
    x_val_ODD, y_val_ODD, w_NN_val_ODD, w_phys_val_ODD = load_ds(os.path.join(cfg['Setup'][data_path], 'ShuffleMerge_ODDmodel_VAL.parquet'),
                                                                    cfg['Features']['train'], cfg['Features']['truth'], cfg['Features']['weight'], eval=True)

    if args.gpu:
        print(f"Storing datasets on GPU")
        # Store datasets on gpu
        x_train_gpu_EVEN = cp.array(x_train_EVEN)
        y_train_gpu_EVEN = cp.array(y_train_EVEN)
        w_train_gpu_EVEN = cp.array(w_train_EVEN)
        x_train_gpu_ODD = cp.array(x_train_ODD)
        y_train_gpu_ODD = cp.array(y_train_ODD)
        w_train_gpu_ODD = cp.array(w_train_ODD)
        x_val_gpu_EVEN = cp.array(x_val_EVEN)
        x_val_gpu_ODD = cp.array(x_val_ODD)
        del x_train_EVEN, y_train_EVEN, w_train_EVEN, x_train_ODD, y_train_ODD, w_train_ODD, x_val_EVEN, x_val_ODD





    main()

