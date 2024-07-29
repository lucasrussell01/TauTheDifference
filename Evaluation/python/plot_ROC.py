from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
import mplhep as hep

plt.style.use(hep.style.ROOT)
purple = (152/255, 152/255, 201/255)
yellow = (243/255,170/255,37/255)
blue = (2/255, 114/255, 187/255)
green = (159/255, 223/255, 132/255)
red = (203/255, 68/255, 10/255)

plt.rcParams.update({"font.size": 14})

# Plot the ROC curves for Higgs Vs Genuine and Higgs Vs Fake

model_dir = "../../Training/python/XGB_Models/BDTClassifier/model_2907"

pred_df = pd.read_parquet(os.path.join(model_dir, 'EVAL_predictions.parquet'))

# extract categories
taus = pred_df.loc[pred_df['class_label'] == 0]
fake = pred_df[pred_df['class_label'] == 2]
higgs = pred_df[(pred_df['class_label'] == 11) | (pred_df['class_label'] == 12)]
ggH = pred_df[pred_df['class_label'] == 11]
VBF = pred_df[pred_df['class_label'] == 12]
total_bkg = pred_df[(pred_df['class_label'] == 0) | (pred_df['class_label'] == 2)]

# higgs and tau comparison
higgstau = pd.concat([higgs, taus])
y_true = higgstau['class_label'].replace({11: 1, 12: 1})
y_pred = higgstau['pred_1']
w = higgstau['weight']

# calculate roc properties
fpr_ht, tpr_ht, thresholds_ht = metrics.roc_curve(y_true, y_pred, sample_weight=w)

# # plot roc curve Higgs vs tau
# fix, ax = plt.subplots(figsize = (6,6))
# ax.grid()
# ax.plot(tpr_ht, fpr_ht, label = "XGBClassifier")
# ax.set_xlim(0,1)
# ax.set_ylim(1e-4,1)
# ax.set_yscale('log')
# ax.set_xlabel('Higgs ID Efficiency')
# ax.set_ylabel('Tau Mis-ID Probability')
# ax.legend()
# ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
# ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
# ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
# plt.savefig(os.path.join(model_dir, 'ROC_higgs_vs_taus.pdf'))
# plt.close()


# higgs and fake comparison
higgsfake = pd.concat([higgs, fake])
y_true = higgsfake['class_label'].replace({11: 1, 12: 1, 2:0})
y_pred = higgsfake['pred_1']
w = higgsfake['weight']

# calculate roc properties
fpr_hf, tpr_hf, thresholds_hf = metrics.roc_curve(y_true, y_pred, sample_weight=w)

# # plot roc curve Higgs vs fake
# fix, ax = plt.subplots(figsize = (6,6))
# ax.grid()
# ax.plot(tpr_hf, fpr_hf, label = "XGBClassifier", color=red)
# ax.set_xlim(0,1)
# ax.set_ylim(1e-4,1)
# ax.set_yscale('log')
# ax.set_xlabel('Higgs ID Efficiency')
# ax.set_ylabel('Fake Mis-ID Probability')
# ax.legend()
# ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
# ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
# ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
# plt.savefig(os.path.join(model_dir, 'ROC_higgs_vs_fake.pdf'))
# plt.close()

# Higgs VS Backgrounds (one plot)
fix, ax = plt.subplots(figsize = (6,6))
ax.grid()
ax.plot(tpr_hf, fpr_hf, label = "XGBClassifier vs Fake", color=red)
ax.plot(tpr_ht, fpr_ht, label = "XGBClassifier vs Tau")
ax.set_xlim(0,1)
ax.set_ylim(1e-4,1)
ax.set_yscale('log')
ax.set_xlabel('Higgs ID Efficiency')
ax.set_ylabel('Mis-ID Probability')
ax.legend()
ax.text(0.7, 1.02, "2022 (13.6 TeV)", fontsize=14, transform=ax.transAxes)
ax.text(0.01, 1.02, 'CMS', fontsize=20, transform=ax.transAxes, fontweight='bold', fontfamily='sans-serif')
ax.text(0.16, 1.02, 'Work in Progress', fontsize=16, transform=ax.transAxes, fontstyle='italic',fontfamily='sans-serif')
plt.savefig(os.path.join(model_dir, 'ROC_higgs_vs_background.pdf'))
plt.close()




