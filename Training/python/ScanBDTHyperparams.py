import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os


# Load training dataset
train_df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_TRAIN.parquet')
x_train = train_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_train = train_df['class_label'].replace({11: 1, 12: 1})
w_train = train_df['NN_weight'] # use class balanced weight
del train_df

# Load validation dataset
val_df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_VAL.parquet')
x_val = val_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_val = val_df['class_label'].replace({11: 1, 12: 1})
w_val = val_df['NN_weight']
del val_df



params = {'max_depth': [3, 4, 5],
              'learning_rate': [0.05, 0.1, 0.2, 0.3],
            #   'reg_lambda': [1, 5, 10], 
            #   'min_child_weight': [1, 5, 10],
              'n_estimators': [100, 200, 300, 500]
              }


print("Begining Hyperparameter Scan")

for d in params['max_depth']:
    for l in params['learning_rate']:
        for n in params['n_estimators']:
        

            model = XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=l,
                                    n_estimators=n, max_depth=d, min_child_weight=1, reg_lambda=1)

            model.fit(x_train, y_train, sample_weight=w_train)

            # Training predictions:
            y_pred_train_labels = model.predict(x_train)
            accuracy_train = accuracy_score(y_train, y_pred_train_labels, sample_weight=w_train)
            # Validation predictions:
            y_pred_val_labels = model.predict(x_val)
            accuracy_val = accuracy_score(y_val, y_pred_val_labels, sample_weight=w_val)
            print(f"Accuracy - Validation: {accuracy_val} (Training: {accuracy_train}) || MD: {d}, LR: {l}, NE: {n}" )
