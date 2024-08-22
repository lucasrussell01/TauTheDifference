import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os


# SHUFFLE MERGE DIRECTORY
sample_path = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt'
train_path = os.path.join(sample_path, 'ShuffleMerge_TRAIN.parquet')
val_path = os.path.join(sample_path, 'ShuffleMerge_VAL.parquet')

# FEATURES TO USEA FOR TRAINING
train_features = ['pt_1', 'pt_2', 'eta_1', 'eta_2', 'phi_1', 'phi_2', 'dR', 'pt_tt', 'pt_tt_met',
            'mt_1', 'mt_2', 'mt_lep', 'mt_tot', 'met_pt', 'met_dphi_1', 'met_dphi_2',
            'dphi', 'm_vis', 'pt_vis', 'n_jets', 'n_bjets', 'mjj', 'jdeta', 'dijetpt',
            'jpt_1', 'jpt_2', 'jeta_1', 'jeta_2',
            'svfitMass',
            'svfitMass_err']
            # 'FastMTT_Mass']


# Load training dataset
train_df = pd.read_parquet(train_path)
x_train = train_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_train = train_df['class_label'].replace({11: 1, 12: 1})
w_train = train_df['NN_weight'] # use class balanced weight
del train_df

# Load validation dataset
val_df = pd.read_parquet(val_path)
x_val = val_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_val = val_df['class_label'].replace({11: 1, 12: 1})
w_val = val_df['NN_weight']
del val_df



params = {'max_depth': [3, 5, 7, 10],
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
            #   'reg_lambda': [1, 5, 10], 
              'min_child_weight': [1, 5, 10],
              'n_estimators': [50, 100, 200, 300, 500, 750]
              }


print("Begining Hyperparameter Scan")


best_accuracy = 0
best_depth = 0
best_lr = 0
best_ne = 0
best_child = 0

for d in params['max_depth']:
    for l in params['learning_rate']:
        for n in params['n_estimators']:
          for c in params['min_child_weight']:
        

              model = XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=l,
                                      n_estimators=n, max_depth=d, min_child_weight=c)

              model.fit(x_train, y_train, sample_weight=w_train)

              # Training predictions:
              y_pred_train_labels = model.predict(x_train)
              accuracy_train = accuracy_score(y_train, y_pred_train_labels, sample_weight=w_train)
              # Validation predictions:
              y_pred_val_labels = model.predict(x_val)
              accuracy_val = accuracy_score(y_val, y_pred_val_labels, sample_weight=w_val)
              if (accuracy_val > best_accuracy) and (abs(accuracy_train - accuracy_val) < 0.03):
                print(f"**** NEW BEST ACCURACY: {accuracy_val} ({accuracy_train} training) ****")
                best_accuracy = accuracy_val
                best_depth = d
                best_lr = l
                best_ne = n
                best_child = c


              print(f"Accuracy - Validation: {accuracy_val} (Training: {accuracy_train}) || MD: {d}, LR: {l}, NE: {n}, MCW: {c}" )


print(f"\n \n Best params are: MD: {best_depth}, LR: {best_lr}, NE: {best_ne}, MCW: {best_child} with cal accuracy: {best_accuracy}")
