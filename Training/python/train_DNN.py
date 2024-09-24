from tf_model_blocks import *
from tensorflow.keras.models import Model
from tf_losses import NNLosses
import os
import yaml
import pandas as pd

def load_ds(path, feat_names, y_name, w_name, eval = False):
    df = pd.read_parquet(path)
    x = df[feat_names]
    y = df[y_name].replace({11: 1, 12: 1})
    w = df[w_name]
    if eval:
        phys_w = df['weight']
        return x, y, w, phys_w
    else:
        return x, y, w


def create_model(features, dropout_rate=0.2):

    # TODO: Architecture could be defined in a config file
    # Architecture
    input_flat = Input(name="input_flat", shape=(len(features)))
    dense_1 = dense_block(input_flat, 50, dropout=dropout_rate, n="_dense_1")
    dense_2 = dense_block(dense_1, 100, dropout=dropout_rate, n="_dense_2")
    dense_3 = dense_block(dense_2, 250, dropout=dropout_rate, n="_dense_3")
    dense_4 = dense_block(dense_3, 250, dropout=dropout_rate, n="_dense_4")
    dense_5 = dense_block(dense_4, 100, dropout=dropout_rate, n="_dense_5")
    dense_6 = dense_block(dense_5, 50, dropout=dropout_rate, n="_dense_6")
    dense_final = dense_block(dense_6, 3, dropout=dropout_rate, n="_dense_final")
    output = Activation("softmax", name="output")(dense_final)

    # Create model
    model = Model(input_flat, output, name="model_test")

    return model

def compile_model(model):

    # TODO: Specify Optimiser via config
    opt = tf.keras.optimizers.Nadam(learning_rate=1e-3)

    # model here
    model.compile(loss=NNLosses.classification_loss, optimizer=opt, metrics=["accuracy"])


def main():
    # Config
    cfg = yaml.safe_load(open("../config/DNNconfig.yaml"))

    # Load training dataset
    train_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_TRAIN.parquet')
    x_train, y_train, w_train = load_ds(train_path, cfg['Features']['train'],
                                        cfg['Features']['truth'], cfg['Features']['weight'])
    y_train_onehot = tf.one_hot(y_train, 3)
    # Load Validation dataaset
    val_path = os.path.join(cfg['Setup']['input_path'], 'ShuffleMerge_VAL.parquet')
    x_val, y_val, w_NN_val, w_phys_val = load_ds(val_path, cfg['Features']['train'],
                                 cfg['Features']['truth'], cfg['Features']['weight'], eval=True)
    y_val_onehot = tf.one_hot(y_val, 3)
    model = create_model(cfg['Features']['train'])
    compile_model(model)

    model.fit(x_train, y_train_onehot, sample_weight=w_train, validation_data=(x_val, y_val_onehot, w_NN_val), epochs=10, batch_size=256)

    # Save model
    save_dir = os.path.join(cfg['Setup']['model_outputs'], cfg['Setup']['model_name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save(f'{save_dir}/model.h5')
    print(f"Training Complete! Model saved to: {save_path} \n")
    # Save features used:
    with open(os.path.join(save_dir, 'train_cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)





if __name__=="__main__":
    main()