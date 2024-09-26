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
    parser.add_argument('--n_trials', type=int, help="Number of trials to attempt")
    parser.add_argument('--study_name', type=str, help="Name of study (can use to resume)")
    return parser.parse_args()


def val_eval(y_pred_proba, x, y, w_phys):
    # Get predictions
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
    ams = AMS(s_counts, bkg_counts)
    ams_tot = np.sqrt(np.sum(ams**2))
    print("AMS Score (bin by bin):", ams)
    print(f"Overall AMS: {ams_tot}")
    return ams_tot


def objective(trial):

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

    model = XGBClassifier(**param)
    model.fit(x_train, y_train, sample_weight=w_train)

    # Inference on validation dataset:
    y_pred_val = model.predict_proba(x_val)
    ams = val_eval(y_pred_val, x_val, y_val, w_val_phys)

    return ams


def main():

    print(f"Optimizing hyperparameters for XGBoost model with {args.n_trials} trials")

    if args.n_trials is not None:

        # Optuna study to optimize hyperparameters
        study = optuna.create_study(direction="maximize", study_name=args.study_name,
                                storage=f"sqlite:///{args.study_name}.db?timeout=10000", load_if_exists=True)
        # Begin search
        study.optimize(objective, n_trials=args.n_trials, n_jobs=-1)

        # Summary
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))



if __name__ == "__main__":
    args = get_args()
    # Load training and validation datasets
    cfg = yaml.safe_load(open("../config/BDTconfig.yaml"))
    train_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_TRAIN.parquet')
    val_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_VAL.parquet')
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])
    x_val, y_val, w_val_NN, w_val_phys = load_ds(val_path, cfg['Features']['train'],
                                            cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    main()

