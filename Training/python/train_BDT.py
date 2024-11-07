import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd
import os
import yaml
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Classifier Training")
    parser.add_argument('--channel', type=str, help="Channel to train", required=True)
    parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    return parser.parse_args()

def load_ds(path, feat_names, y_name, w_name, eval = False):
    df = pd.read_parquet(path)
    x = df[feat_names]
    y = df[y_name]
    w = df[w_name]
    if eval:
        phys_w = df['weight']
        return x, y, w, phys_w
    else:
        return x, y, w

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def validation(model, cfg, parity):
    print(f'\033[1mModel performance: \033[0m')
    val_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_VAL.parquet')
    x, y, w_NN, w_phys = load_ds(val_path, cfg['Features']['train'],
                                 cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    # Get predictions
    y_pred_proba = model.predict_proba(x) # raw score
    y_pred = y_pred_proba.argmax(axis=1) # predicted label
    # Accuracy
    accuracy = accuracy_score(y, y_pred, sample_weight=w_NN)
    print("Validation Accuracy:", accuracy)
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
    print("AMS Score (bin by bin):", ams)
    print(f"\033[1;32mAMS: {np.sqrt(np.sum(ams**2))} \033[0m")
    # AUC Score
    truth = y_higgs.replace({2:0, 0:0}) # binary Higgs vs all
    auc = roc_auc_score(truth, y_pred_higgs, sample_weight=w_pred_higgs)
    print("AUC Score:", auc)
    del x, y, w_NN, w_phys

def train_model(cfg, parity):
    # Input path (depends on even/odd)
    train_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_TRAIN.parquet')

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])

    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**cfg['param'])
    model.fit(x_train, y_train, sample_weight=w_train)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_dir_name'], parity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{cfg['Setup']['model_prefix']}_{parity}.json")
    model.save_model(save_path)
    # Get Training accuracy
    y_pred_labels = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred_labels, sample_weight=w_train)
    print(f"Training Complete! (accuracy: {accuracy}) - Model saved to: {save_path}")
    # Save features used:
    with open(os.path.join(save_dir, 'train_cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    del x_train, y_train, w_train

    return model



def main():
    args = get_args()
    if args.channel == 'tt': # Fully hadronic has different vsjet cuts
        if args.cut == "tight":
            print("Training for tt channel (TIGHT Vsjet cut)")
            cfg = yaml.safe_load(open("../config/tt/BDTconfig_tight.yaml"))
            cfg['Setup']['model_prefix'] = 'model_tight'
        elif args.cut == "vtight":
            print("Training for tt channel (VTIGHT Vsjet cut)")
            cfg = yaml.safe_load(open("../config/tt/BDTconfig_vtight.yaml"))
            cfg['Setup']['model_prefix'] = 'model_vtight'
        else: # use medium by default
            print("Training for tt channel (MEDIUM Vsjet cut)")
            cfg = yaml.safe_load(open("../config/tt/BDTconfig_medium.yaml"))
            cfg['Setup']['model_prefix'] = 'model_medium' # begining of model json name (add parity after)
    elif args.channel == 'mt':
        print("Training for MuTau channel")
        cfg = yaml.safe_load(open("../config/mt/BDTconfig.yaml"))
        cfg['Setup']['model_prefix'] = 'model'

    # Train the model to be applied on EVEN events
    model = train_model(cfg, 'EVEN')
    validation(model, cfg, 'EVEN')
    print('---------------------------------- \n')

    # Train the model to be applied on ODD events
    model = train_model(cfg, 'ODD')
    validation(model, cfg, 'ODD')
    print('---------------------------------- \n')

if __name__ == "__main__":
    main()
