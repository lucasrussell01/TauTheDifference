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
    parser.add_argument('--study_name', type=str, help="Name of study (can use to resume)")
    parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    return parser.parse_args()

def validation(model, cfg, parity, ds_path='input_path'):
    # Load validation dataset
    val_path = os.path.join(cfg['Setup'][ds_path], f'ShuffleMerge_{parity}model_VAL.parquet')
    x, y, w_NN, w_phys = load_ds(val_path, cfg['Features']['train'],
                                 cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    # Get predictions
    y_pred_proba = model.predict_proba(x) # raw score
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


def train_model(cfg, parity, param, ds_path='input_path'):
    # Input path (depends on even/odd)
    train_path = os.path.join(cfg['Setup'][ds_path], f'ShuffleMerge_{parity}model_TRAIN.parquet')

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])

    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**param)
    model.fit(x_train, y_train, sample_weight=w_train)

    del x_train, y_train, w_train

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



    # Find correct dataset to use (from channel and cut argument)
    if args.channel == 'tt':
        # Load training config
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


    # Train even model
    model_even = train_model(cfg, "EVEN", param, data_path)
    ams_even = validation(model_even, cfg, "EVEN", data_path)

    # Train odd model
    model_odd = train_model(cfg, "ODD", param, data_path)
    ams_odd = validation(model_odd, cfg, "ODD", data_path)

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
        study.optimize(objective, n_trials=args.n_trials, n_jobs=4)

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
    main()

