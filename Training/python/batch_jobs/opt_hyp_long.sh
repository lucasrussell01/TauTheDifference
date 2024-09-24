#!/bin/bash
echo "Begining job"
source ~/.bashrc
echo "Sourced bashrc"
mamba activate tau-ml-old
echo "Activated environment"
cd /vols/cms/lcr119/offline/HiggsCP/SignalClassifier/HiggsTauTauClassifier/Training/python
echo "Ready to run hyperparameter optimization"
python searchBDTparams.py --n_trials=500 --study_name=batched_500trials
echo "Finished"
