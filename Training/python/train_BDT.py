import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

def load_ds(path, feat_names, y_name, w_name, eval = False):
    df = pd.read_parquet(path)
    x = df[feat_names]
    y = df[y_name].replace({11: 1, 12: 1})
    w = df[w_name]
    if eval:
        phys_w = df['weight']
        return x, y, w, phys_w
    else:
        return x, y, w

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def val_eval(model, cfg):
    val_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_VAL.parquet')
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
    print(f"Overall AMS: {np.sqrt(np.sum(ams**2))}")
    del x, y, w_NN, w_phys

def train_model(cfg):
    # Input path
    train_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_TRAIN.parquet')

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])

    # Model training
    print("Training XGBClassifier model")
    model = XGBClassifier(**cfg['param'])
    model.fit(x_train, y_train, sample_weight=w_train)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.json')
    model.save_model(save_path)
    print(f"Training Complete! Model saved to: {save_path} \n")
    # Get Training accuracy
    y_pred_labels = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred_labels, sample_weight=w_train)
    print("Training Accuracy:", accuracy)
    # Save features used:
    with open(os.path.join(save_dir, 'train_cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    del x_train, y_train, w_train

    return model



def main():
    cfg = yaml.safe_load(open("../config/BDTconfig.yaml"))
    model = train_model(cfg)
    val_eval(model, cfg)

if __name__ == "__main__":
    main()
