import matplotlib.pyplot as plt
import pandas as pd
import os
import mplhep as hep
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import seaborn as sn
import yaml


def plot_confusion_matrix(cfg, parity):

    lumi = 61.90

    sn.set(font_scale=1.4)

    plt.style.use(hep.style.ROOT)
    purple = (152/255, 152/255, 201/255)
    yellow = (243/255,170/255,37/255)
    blue = (2/255, 114/255, 187/255)
    green = (159/255, 223/255, 132/255)
    red = (203/255, 68/255, 10/255)

    plt.rcParams.update({"font.size": 14})

    model_dir = os.path.join(cfg['model_path'], cfg['model_name'], parity)

    pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))

    y_pred = pred_df['pred_label'] # take max pred label
    y_true = pred_df['class_label'].replace({11: 1, 12: 1})
    weights = pred_df['weight']

    labels = ["Tau", "Higgs", "Fake"]

    # Efficiency confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true', sample_weight=weights)
    fig, ax = plt.subplots(figsize=(8,6.4))
    plt.axhline(y = 0, color='k',linewidth = 3)
    plt.axhline(y = 3, color = 'k', linewidth = 3)
    plt.axvline(x = 0, color = 'k',linewidth = 3)
    plt.axvline(x = 3, color = 'k', linewidth = 3)
    sn.heatmap(cm, annot=True, cmap='Blues', xticklabels =  labels, yticklabels = labels, annot_kws={"fontsize":12})
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    ax.text(1.02, 0.45, "Efficiency", fontsize=12, transform=ax.transAxes, rotation=90)
    ax.text(0.6, 1.02, rf"{lumi:.2f} fb$^{{-1}}$ (13.6 TeV)", fontsize=14, transform=ax.transAxes)
    ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.text(0.15, 1.02, 'Work in Progress', fontsize=14, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
    plt.savefig(os.path.join(model_dir, 'plots', 'Efficiency_CM.pdf'))

    # Purity confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='pred', sample_weight=weights)
    fig, ax = plt.subplots(figsize=(8,6.4))
    plt.axhline(y = 0, color='k',linewidth = 3)
    plt.axhline(y = 3, color = 'k', linewidth = 3)
    plt.axvline(x = 0, color = 'k',linewidth = 3)
    plt.axvline(x = 3, color = 'k', linewidth = 3)
    sn.heatmap(cm, annot=True, cmap='Blues', xticklabels =  labels, yticklabels = labels, annot_kws={"fontsize":12})
    plt.ylabel("True Category")
    plt.xlabel("Predicted Category")
    ax.text(1.02, 0.455, "Purity", fontsize=12, transform=ax.transAxes, rotation=90)
    ax.text(0.6, 1.02, rf"{lumi:.2f} fb$^{{-1}}$ (13.6 TeV)", fontsize=14, transform=ax.transAxes)
    ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
    ax.text(0.15, 1.02, 'Work in Progress', fontsize=14, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
    plt.savefig(os.path.join(model_dir, 'plots', 'Purity_CM.pdf'))

    print("Confusion matrices produced for ")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("../config/config.yaml"))
    plot_confusion_matrix(cfg, 'EVEN')
    plot_confusion_matrix(cfg, 'ODD')
    print("Evaluation complete!")