import pandas as pd
import numpy as np


df = pd.read_parquet('/vols/cms/lcr119/offline/HiggsCP/data/earlyrun3/SM_noPU/tt/ShuffleMerge_ALL.parquet')

run3_22_df = df[df['era']==0]
run3_22EE_df = df[df['era']==1]



def analyse_fracs(df):
    total = len(df)
    counts = df['class_label'].value_counts()
    print(f"Number of ggH events: {counts[11]} ({100*counts[11]/total:.2f}%)")
    print(f"Number of VBF events: {counts[12]} ({100*counts[12]/total:.2f}%)")
    print(f"Number of DY events: {counts[0]} ({100*counts[0]/total:.2f}%)")
    print(f"Number of QCD events: {counts[2]} ({100*counts[2]/total:.2f}%)")


def analyse_weights(df):
    total = np.sum(df['weight'])
    # Sum existing physics weights accross each category
    w_sum_ggH = df.loc[df['class_label'] == 11, 'weight'].sum()
    w_sum_VBF = df.loc[df['class_label'] == 12, 'weight'].sum()
    w_sum_bkg = df.loc[df['class_label'] == 2, 'weight'].sum()
    w_sum_taus = df.loc[df['class_label'] == 0, 'weight'].sum()
    print(f"Sum of ggH weights: {w_sum_ggH:.2f} ({100*w_sum_ggH/total:.2f}%)")
    print(f"Sum of VBF weights: {w_sum_VBF:.2f} ({100*w_sum_VBF/total:.2f}%)")
    print(f"Sum of DY weights: {w_sum_taus:.2f} ({100*w_sum_taus/total:.2f}%)")
    print(f"Sum of QCD weights: {w_sum_bkg:.2f} ({100*w_sum_bkg/total:.2f}%)")




print(f"RUN 3 2022 [Total events: {len(run3_22_df)}]")

# analyse_fracs(run3_22_df)
analyse_weights(run3_22_df)
print("-----------------------")


print(f"RUN 3 2022EE [Total events: {len(run3_22EE_df)}]")

# analyse_fracs(run3_22EE_df)
analyse_weights(run3_22EE_df)

print("-----------------------")


