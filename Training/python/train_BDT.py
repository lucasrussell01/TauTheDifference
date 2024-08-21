import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

# SHUFFLE MERGE DIRECTORY
sample_path = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt'
train_path = os.path.join(sample_path, 'ShuffleMerge_TRAIN.parquet')
val_path = os.path.join(sample_path, 'ShuffleMerge_VAL.parquet')

# MODEL SAVE DIRECTORY
save_dir = 'XGB_Models/BDTClassifier/model_svfit/'

# FEATURES TO USEA FOR TRAINING
train_features = ['pt_1', 'pt_2', 'eta_1', 'eta_2', 'phi_1', 'phi_2', 'dR', 'pt_tt', 'pt_tt_met',
            'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'met_pt', 'met_dphi_1', 'met_dphi_2',
            'dphi', 'm_vis', 'pt_vis', 'n_jets', 'n_bjets', 'mjj', 'jdeta', 'dijetpt',
            'jpt_1', 'jpt_2', 'jeta_1', 'jeta_2',
            'svfitMass']#,
            # 'FastMTT_Mass']#,
            # 'svfitMass_err']

# Load training dataset
train_df = pd.read_parquet(train_path)
x_train = train_df[train_features]
y_train = train_df['class_label'].replace({11: 1, 12: 1})
w_train = train_df['NN_weight'] # use class balanced weight
del train_df


if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model.json')

# Model training 
# TODO: Add hyperparams to config or similar

print("Training model")
               

# model = XGBClassifier(objective='multi:softmax', num_class=3)

         
model = XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=0.1,
                        n_estimators=200, max_depth=3, min_child_weight=1, reg_lambda=1)

model.fit(x_train, y_train, sample_weight=w_train)

model.save_model(save_path) # TODO: Add a run hash or similar

print(f"Training Complete! Model saved to: {save_path}")


# Training predictions:
y_pred_train = model.predict_proba(x_train)
y_pred_train_labels = y_pred_train.argmax(axis=1)
accuracy_train = accuracy_score(y_train, y_pred_train_labels, sample_weight=w_train)
print("Training Accuracy:", accuracy_train)

# clear memory
del x_train, y_train, w_train

# Load validation dataset
val_df = pd.read_parquet(val_path)
x_val = val_df[train_features]
y_val = val_df['class_label'].replace({11: 1, 12: 1})
w_val = val_df['NN_weight']
del val_df

# Validation predictions:
y_pred_val = model.predict_proba(x_val)
y_pred_val_labels = y_pred_val.argmax(axis=1)
accuracy_val = accuracy_score(y_val, y_pred_val_labels, sample_weight=w_val)
print("Validation Accuracy:", accuracy_val)
