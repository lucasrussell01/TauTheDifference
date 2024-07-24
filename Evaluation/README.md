# Evaluation

This directory contains tools to analyse trained models/predictions.

## Apply Training

Saved `XGBClassifier` models should be applied to the evaluation dataset with
`apply_BDTtraining.py`.

This will store predictions, labels and weights in the model output directory.

## Plotting

`plot_scores.py` created histograms of the score distributions for each of the three (Higgs, Genuine and Fake) categories.

`plot_roc.py` creates ROC curves of the model performance.