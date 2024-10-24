import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml
import tensorflow as tf


def classification_loss(target, output):
    loss = tf.keras.losses.categorical_crossentropy(target, output)

cfg = yaml.safe_load(open("../config/config.yaml"))
train_cfg = yaml.safe_load(open(os.path.join("../../Training/python/XGB_Models/HyperOpt", "model_best100LR", 'train_cfg.yaml')))
feature_cfg = train_cfg['Features']
input_path = os.path.join(train_cfg['Setup']['input_path'], "ShuffleMerge_EVAL.parquet")

# Load evaluation dataset
eval_df = pd.read_parquet(input_path)
x_eval = eval_df[feature_cfg['train']]
y_eval_raw = eval_df[feature_cfg['truth']] # keep separate higgs labels
y_eval = y_eval_raw.replace({11: 1, 12: 1})
w_eval = eval_df[feature_cfg['weight']] #  NN weight
w_plot = eval_df['weight'] # NOT the NN weight (normalisation removed)

print(f"\nApplying Training for model: {cfg['model_name']}")

# Load trained model
model_dir = os.path.join(cfg['model_path'], cfg['model_name'])
model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'), custom_objects={'classification_loss': classification_loss})


y_pred_eval = model.predict(x_eval)
y_pred_eval_labels = y_pred_eval.argmax(axis=1)
accuracy_eval = accuracy_score(y_eval, y_pred_eval_labels, sample_weight=w_eval)
print("Evaluation Accuracy:", accuracy_eval)


# store predictions as a new df
df_res = pd.DataFrame()
df_res['class_label'] = y_eval_raw
df_res['pred_0'] = y_pred_eval[:, 0]
df_res['pred_1'] = y_pred_eval[:, 1]
df_res['pred_2'] = y_pred_eval[:, 2]
df_res['pred_label'] = y_pred_eval_labels # max proba label
df_res['weight'] = w_plot
df_res['NN_weight'] = w_eval
df_res['unit_weight'] = 1 # unit weight
df_res.to_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))

print(f"Evaluation of {model_dir.split('/')[-1]} complete!")
