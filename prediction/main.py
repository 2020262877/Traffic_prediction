import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np  # linear algebra
from scipy.stats import spearmanr, pearsonr
import pmdarima as pm
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import StandardScaler
# import os  # accessing directory structure
# import altair as alt
import tqdm

import naive_correlation_Analysis as tfe
import baselineModel as bm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv(r'D:\毕业设计\github代码\1在看\preprocessed-datasets\processed_dataset.csv')
    bm.main(df)
    # tfe.fun1(df)
    # tfe.fun2(df)
    # tfe.fun3(df)

