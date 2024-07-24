# Training

## BDT Classifier

`XGBClassifier` is used to separate events into three classes (`11` and `12` are relabeled to `1` so there is just one Higgs class).

`train_BDT.py` is the main code to run training and save the model.

`ScanBDTHyperparams.py`can be used to find optimal hyperparameters.