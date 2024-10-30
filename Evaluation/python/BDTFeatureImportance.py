import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt

def feature_study(cfg, parity):
    print(f"Plotting features importance for model: {cfg['model_name']} - {parity}")
    # parity is EVEN or ODD - the event parity the model is applied to
    train_cfg = yaml.safe_load(open(os.path.join(cfg['model_path'],  cfg['model_name'], parity, 'train_cfg.yaml')))
    feature_cfg = train_cfg['Features']

    # Load trained model
    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)
    model = XGBClassifier()
    model.load_model(os.path.join(model_dir, f'{train_cfg["Setup"]["model_prefix"]}_{parity}.json'))

    # make folder for feature importance plots
    if not os.path.exists(os.path.join(model_dir, 'features')):
        os.makedirs(os.path.join(model_dir, 'features'))

    # Plot Feature Importance
    for imp_type in ['gain', 'cover', 'weight']:
        fig, ax = plt.subplots()
        xgb.plot_importance(model, importance_type=imp_type, ax=ax, title=f'Feature Importance ({imp_type})', show_values=False, grid=False)
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'features', f'importance_{imp_type}.pdf'))
        print(f"Plotted {imp_type} feature importance")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/config.yaml"))
    feature_study(cfg, 'EVEN')
    feature_study(cfg, 'ODD')


