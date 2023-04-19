import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ensemble_test import clear_plt, get_plots


def run():
    criterion_df = pd.read_csv('../data/misc/criterion_df.csv')
    for col in criterion_df.columns:
        if col.startswith('astat'):
            plt.plot(criterion_df[col])
    plt.xlabel('epoch/50')
    plt.ylabel('anderson stat')
    plt.savefig('../data/misc/astat_plot.png')
    clear_plt()


if __name__=='__main__':
    run()