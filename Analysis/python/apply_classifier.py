import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml



# APPLY CLASSIFIER MODEL TO HIGGSDNA OUTPUTS


def apply_classifier_model(subdirectory):
    # Load XGB model
    logger.info(f"xgb version: {xgb.__version__}")
    model = XGBClassifier() # use 2022 model for now
    model.load_model('/vols/cms/lcr119/offline/HiggsCP/HiggsDNA/scripts/ditau/config/SignalBkgClassifier/model.json')
    # Load model features
    features = yaml.safe_load(open('scripts/ditau/config/SignalBkgClassifier/train_cfg.yaml'))['Features']['train']
    logger.info(f"Model successfully loaded")
    # Find merged parquet file to evaluate
    merged_file =os.path.join(directory, "merged.parquet")
    if not os.path.exists(merged_file):
        logger.info(f"Merged file does not exist!")
        return
    df = pd.read_parquet(merged_file)
    # apply model using relevant features
    y_pred = model.predict_proba(df[features])
    logger.info(f"Model successfully applied")
    # add classifier scores to the dataframe
    df['BDT_pred_score'] = np.max(y_pred, axis=1) # score of highest class
    df['BDT_pred_class'] = y_pred.argmax(axis=1) # class with highest score
    # individual scores
    df['BDT_raw_score_tau'] = y_pred[:, 0]
    df['BDT_raw_score_higgs'] = y_pred[:, 1]
    df['BDT_raw_score_fake'] = y_pred[:, 2]
    # write out the file with classifier scores
    df.to_parquet(merged_file)
    logger.info(f"Classifier scores added to merged file")


def main():

    # apply model to subdirectory of HiggsDNA output which contains merged.parquet file
    apply_classifier_model(subdirectory)