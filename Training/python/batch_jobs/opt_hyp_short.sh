#!/bin/bash
echo "Begining job"
source ~/.bashrc
echo "Sourced bashrc"
mamba activate tau-ml-old
echo "Activated environment"
cd /vols/cms/lcr119/offline/HiggsCP/SignalClassifier/HiggsTauTauClassifier/Training/python
echo "Ready to run hyperparameter optimization"
python searchBDTparams.py --n_trials=100 --study_name=batched_100trials
echo "Finished"
