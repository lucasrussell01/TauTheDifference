import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
import argparse
import mplhep as hep

plt.style.use(hep.style.ROOT)
plt.rcParams.update({"font.size": 24})


def get_args():
    parser = argparse.ArgumentParser(description="XGBoost Classifier Evaluation")
    parser.add_argument('--channel', type=str, help="Channel to evaluate", required=True)
    return parser.parse_args()

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
        xgb.plot_importance(model, importance_type=imp_type, ax=ax, title=f'', show_values=False, grid=False)
        ax.set_xlabel(rf'Feature Score')
        ax.set_ylabel('')
        ax.text(0.8, 0.05, f'{imp_type.upper()}', fontsize=24, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'features', f'importance_{imp_type}.pdf'))
        print(f"Plotted {imp_type} feature importance")

if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/config_FFs.yaml"))
    # find correct channel
    args = get_args()
    if args.channel == 'tt':
        print("Evaluating for TauTau channel")
        cfg = cfg['tt']
    elif args.channel == 'mt':
        print("Evaluating for MuTau channel")
        cfg = cfg['mt']
    elif args.channel == 'et':
        print("Evaluating for ETau channel")
        cfg = cfg['et']
    # run feature importance
    feature_study(cfg, 'EVEN')
    feature_study(cfg, 'ODD')


