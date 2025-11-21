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
    # parser.add_argument('--cut', type=str, help="VSjet cut to be used", required=False)
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")
    return parser.parse_args()

def load_ds(path, feat_names, y_name, w_name, eval = False):
    df = pd.read_parquet(path)
    if eval:
        x = df[feat_names]
        y = df[y_name]
        w = df[w_name]
        phys_w = df['weight']
        return x, y, w, phys_w
    else:
        df = df[df['weight']>0] # only positive weights can be used for training
        x = df[feat_names]
        y = df[y_name]
        w = df[w_name]
        return x, y, w

def AMS(S, B, b0=0):
    ams = np.sqrt(2*((S+B+b0)*np.log(1+S/(B+b0))-S))
    return ams

def validation(model, cfg, parity, gpu=False):
    print(f'\033[1mModel performance: \033[0m')
    val_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_VAL.parquet')
    x, y, w_NN, w_phys = load_ds(val_path, cfg['Features']['train'],
                                 cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    if gpu:
        print(f"Using GPU for validation")
        x_gpu = cp.array(x)
        # Get predictions
        y_pred_proba = model.predict_proba(x_gpu) # raw score
    else:
        y_pred_proba = model.predict_proba(x) # raw score
    y_pred = y_pred_proba.argmax(axis=1) # predicted label
    # Accuracy
    accuracy = accuracy_score(y, y_pred, sample_weight=w_NN)
    print("Validation Accuracy:", accuracy)
    del x, y, w_NN, w_phys

def train_model(cfg, parity, gpu=False):
    # Input path (depends on even/odd)
    train_path = os.path.join(cfg['Setup']['input_path'], f'ShuffleMerge_{parity}model_TRAIN.parquet')

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])

    # Model training
    print(f"Training XGBClassifier model for \033[1;34m{parity}\033[0m events")
    model = XGBClassifier(**cfg['param'])

    if gpu:
        print(f"Using GPU for training")
        # Store datasets on gpu
        x_gpu = cp.array(x_train)
        y_gpu = cp.array(y_train)
        w_gpu = cp.array(w_train)
        model.fit(x_gpu, y_gpu, sample_weight=w_gpu)
    else:
        model.fit(x_train, y_train, sample_weight=w_train)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_dir_name'], parity)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{cfg['Setup']['model_prefix']}_{parity}.json")
    model.save_model(save_path)
    # Get Training accuracy
    if gpu:
        y_pred_labels = model.predict(x_gpu)
    else:
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
    if args.channel == 'mt':
        print("Training for MuTau channel")
        cfg = yaml.safe_load(open("../config/mt/BDTconfig_FFs.yaml"))
        cfg['Setup']['model_prefix'] = 'model'
    elif args.channel == 'et':
        print("Training for ETau channel")
        cfg = yaml.safe_load(open("../config/et/BDTconfig_FFs.yaml"))
        cfg['Setup']['model_prefix'] = 'model'

    # gpu setup
    if args.gpu:
        import cupy as cp
        cfg['param']['device'] = "gpu" # Use GPU for training

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
