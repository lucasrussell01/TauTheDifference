import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
import os
import numpy as np
plt.style.use(hep.style.ROOT)
purple = (152/255, 152/255, 201/255)
plt.rcParams.update({"font.size": 14})

# path = '/vols/cms/lcr119/offline/HiggsCP/data/ShuffleMerge/2022/tt/ShuffleMerge_ALL.parquet'
path = '/vols/cms/lcr119/offline/HiggsCP/data/raw/2022/tt/DYto2L_M-50_madgraphMLM/nominal/merged.parquet'

# plot histograms for the following columns
columns = ['rawDeepTau2018v2p5VSjet_1', 'rawDeepTau2018v2p5VSjet_2']

if not os.path.exists('tmp'):
    os.makedirs('tmp')

df = pd.read_parquet(path)

for c in df.columns:
    print(c)

print(np.unique(df["idDeepTau2018v2p5VSjet_1"]))

for c in columns:
    mean = np.mean(df[c])
    std = np.std(df[c])
    print(f"Column: [{c}] - Mean = {mean:.2f}, Std = {std:.2f} (will plot up to 5 stds above mean)")
    bin_min = np.min(df[c])
    bin_max = np.min([np.max(df[c]), mean + 5*std])
    bins = np.linspace(bin_min, bin_max, num=25)
    fig, ax = plt.subplots()
    ax.hist(df[c], bins=bins, color=purple)
    plt.xlabel(c)
    plt.savefig(f'tmp/{c}.pdf')