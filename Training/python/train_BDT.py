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

save_dir = 'XGB_Models/BDTClassifier/model_2307/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'model.json')

# Model training 
# TODO: Add hyperparams to config or similar

print("Training model")
               

model = XGBClassifier(objective='multi:softmax', num_class=3)

         
# model = XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=0.1,
                        # n_estimators=200, max_depth=3, min_child_weight=1, reg_lambda=1)

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
val_df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_VAL.parquet')
x_val = val_df.drop(columns=['class_label', 'weight', 'NN_weight'])
y_val = val_df['class_label'].replace({11: 1, 12: 1})
w_val = val_df['NN_weight']
del val_df

# Validation predictions:
y_pred_val = model.predict_proba(x_val)
y_pred_val_labels = y_pred_val.argmax(axis=1)
accuracy_val = accuracy_score(y_val, y_pred_val_labels, sample_weight=w_val)
print("Validation Accuracy:", accuracy_val)
