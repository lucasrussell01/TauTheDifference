import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml


def load_ds(path, feat_names, y_name, w_name):
    df = pd.read_parquet(path)
    x = df[feat_names]
    y = df[y_name].replace({11: 1, 12: 1})
    w = df[w_name]
    return x, y, w


def train_model(cfg):
    # Input path
    train_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_TRAIN.parquet')

    # Load training dataset
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])

    # Model training
    print("Training model")
    model = XGBClassifier(objective='multi:softmax', num_class=3, learning_rate=0.05,
                            n_estimators=750, max_depth=3, min_child_weight=1, reg_lambda=1)
    model.fit(x_train, y_train, sample_weight=w_train)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'model.json')
    model.save_model(save_path)
    print(f"Training Complete! Model saved to: {save_path} \n")
    # Get Training accuracy
    accuracy = predict_acc(model, x_train, y_train, w_train)
    print("Training Accuracy:", accuracy)
    # Save features used:
    with open(os.path.join(save_dir, 'features.yaml'), 'w') as f:
        yaml.dump(cfg['Features'], f)
    del x_train, y_train, w_train

    return model

def predict_acc(model, x, y, w):
    y_pred = model.predict_proba(x)
    y_pred_labels = y_pred.argmax(axis=1)
    accuracy = accuracy_score(y, y_pred_labels, sample_weight=w)
    return accuracy

def val_acc(model, cfg):
    val_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_VAL.parquet')
    x_val, y_val, w_val = load_ds(val_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])
    accuracy_val = predict_acc(model, x_val, y_val, w_val)
    print("Validation Accuracy:", accuracy_val)
    del x_val, y_val, w_val



def main():
    cfg = yaml.safe_load(open("../config/BDTconfig.yaml"))
    model = train_model(cfg)
    val_acc(model, cfg)

if __name__ == "__main__":
    main()
