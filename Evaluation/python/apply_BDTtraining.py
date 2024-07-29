import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os


# Load evaluation dataset
eval_df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_EVAL.parquet')
x_eval = eval_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_eval_raw = eval_df['class_label'] # keep separate higgs labels
y_eval = y_eval_raw.replace({11: 1, 12: 1})
w_eval = eval_df['NN_weight'] #  NN weight
w_plot = eval_df['weight'] # NOT the NN weight (normalisation removed)

# Load trained model
model_dir = "../../Training/python/XGB_Models/BDTClassifier/model_2907"
model = XGBClassifier()
model.load_model(os.path.join(model_dir, 'model.json'))

y_pred_eval = model.predict_proba(x_eval)
y_pred_eval_labels = y_pred_eval.argmax(axis=1)
accuracy_eval = accuracy_score(y_eval, y_pred_eval_labels, sample_weight=w_eval)
print("Evaluation Accuracy:", accuracy_eval)


# store predictions as a new df
df_res = pd.DataFrame()
df_res['class_label'] = y_eval_raw
df_res['pred_0'] = y_pred_eval[:, 0]
df_res['pred_1'] = y_pred_eval[:, 1]
df_res['pred_2'] = y_pred_eval[:, 2]
df_res['weight'] = w_plot
df_res['NN_weight'] = w_eval
df_res.to_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))

print(f"Evaluation of {model_dir.split('/')[-1]} complete!")
