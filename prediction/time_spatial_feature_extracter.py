import numpy
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import numpy as np  # linear algebra
import warnings
from scipy.stats import spearmanr, pearsonr
import pmdarima as pm
from matplotlib import pyplot as plt
from scipy import stats
from datetime import datetime, timedelta
import time
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import StandardScaler
# import os  # accessing directory structure
# import altair as alt
import tqdm

sns.set_context('paper')
# 去除部分warning
warnings.filterwarnings('ignore')

# pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)


def main(df):
    # df.info()
    df = pd.read_csv(df)
    city_name = 'Los Angeles'
    # counts = df.groupby('grid_2000m').size().sort_values(ascending=False)
    #
    # plt.title('Traffic Accidents in Each Grid of {} in (2016-2020) (2km)'.format(city_name), color='grey')
    # plt.xlabel('Grids', color='grey')
    # plt.ylabel('Traffic Accidents Amounts', color='grey')
    # sns.boxplot(data=counts)
    # plt.show()
    # print(counts)  # 11S LT  41 33    1675

    '''# 事故持续时间
    plt.title('Traffic Accidents time duration {} in (2016-2020)'.format(city_name), color='grey')
    plt.xlabel('Time Duration', color='grey')
    plt.ylabel('Traffic Accidents Amounts', color='grey')
    dff = df['Time_Duration']
    sns.boxplot(dff[dff < 500])
    s = pd.Series(dff)
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.5)
    q3 = s.quantile(0.75)
    print('Mean: {}\tQ1: {}\tQ2: {}\tQ3: {}\tMax: {}\tMin:{}\t'.format(s.median(), q1, q2, q3, s.max(), s.min()))

    iqr = q3 - q1
    lowest = max(0, q1 - 1.5 * iqr)
    highest = q3 + 1.5 * iqr
    print('normal range:({}, {})'.format(lowest, highest))
    df[df.Time_Duration < 840].info()
    plt.show()'''

    # 城市交通事故分布图
    plt.title('Traffic Accident Severity Distribution of {} in 2020'.format(city_name), color='grey')
    plt.xlabel('Longitude', color='grey')
    plt.ylabel('Latitude', color='grey')
    sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df[df.Year == 2020], hue='grid_3000m', legend=False, marker='.')
    plt.show()

    # freq = df[df.grid_2000m == '11S LT  41 33'].sort_values(by='Start_Time') \
    #     .groupby(by=[pd.to_datetime(df.Start_Time).map(lambda x: (x.year, x.month, x.day))]) \
    #     .size().sort_values(ascending=True)
    # ----------------------弃用该分析----------------------
    # print(freq)
    # sns.histplot()
    # plt.title('Traffic Accidents occur in 11S LT  41 33 of {} '.format(city_name), color='grey')
    # plt.xlabel('Date', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(freq, bins=7, kde=False, color='b')
    # plt.show()

    # plt.title('Traffic Accident Severity Distribution of {} 11S LT  41 33'.format(city_name), color='grey')
    # plt.xlabel('Longitude', color='grey')
    # plt.ylabel('Latitude', color='grey')
    # sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df[df.grid_2000m == '11S LT  41 33'],
    #                 hue='Severity', legend=False)
    # plt.show()

    # print(df[df.grid_2000m == '11S LT  41 33'].info())
    # df['Week_Num'] = df['Weekday'].map({'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7})
    # plt.title('Accident cases in each week of grid'.format('11S LT  41 33'), color='grey')
    # plt.xlabel('Weekday', color='grey')
    # plt.ylabel('Hour', color='grey')
    # sns.violinplot(x='Weekday', y='Hour',
    #                data=df[df.grid_2000m == '11S LT  41 33'].sort_values(by='Week_Num'))
    # sns.swarmplot(x='Weekday', y='Hour', data=df[df.grid_2000m == '11S LT  41 33'].sort_values(by='Week_Num'),
    #               color="w",  marker='.', c='grey')
    # plt.show()

    # plt.title('Accident cases for a day of {}'.format('11S LT  41 33'), color='grey')
    # plt.xlabel('Hour', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(df[df.grid_2000m == '11S LT  41 33'].Hour, bins=24, kde=True, color='lightcoral')
    # plt.show()

    # plt.title('Accidents per day of the grid cell with maximum number of accident records of {} in (2016-2020)'.format(
    #     city_name), color='grey')
    # plt.xlabel('Months', color='grey')
    # plt.ylabel('Traffic Accidents Amounts', color='grey')
    # sns.displot(freq, kde=True)
    # plt.show()
    #
    # temp_df = df[df.grid_2000m == '11S LT  41 33'].sort_values(by='Start_Time').Severity.to_frame().reset_index(
    #     drop=True).values
    #
    # data, labels = univariate_data(temp_df, 0, len(temp_df) - 1, 5, 1)
    #
    # print(temp_df.shape)
    # print(data.shape)
    # print(labels.shape)
    #
    # for i in range(5):
    #     corr = spearmanr(data[:, i, :].flatten(), labels.flatten())
    #     print(corr)
    #
    # x = pm.c(*df[df.grid_2000m == '11S LT  41 33'].sort_values(by='Start_Time').Severity)
    # pm.acf(x)
    # pm.plot_acf(x)


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def cluster_process(df, day=5):
    df = pd.read_csv(df)
    k = 5
    df = df[['Start_Lat', 'Start_Lng', 'Year', 'Month', 'Day']]
    city_name = 'Los Angeles'

    df = df[df.Year == 2020]
    # df = df[df.Month == 'Oct']
    loc = []
    for i, row in df.iterrows():
        loc.append([float(row['Start_Lat']), float(row['Start_Lng'])])
    print(loc)
    data = whiten(loc)
    print(data)

    codebook, distortion = kmeans(data, k)  # 这个Kmeans好像只返回聚类中心、观测值和聚类中心之间的失真
    plt.title('Traffic Accidents occur in 2020 of {} '.format(city_name), color='grey')
    plt.ylabel('Latitude', color='grey')
    plt.xlabel('Longitude', color='grey')
    plt.scatter(x=data[:, 1], y=data[:, 0], c='g', marker='.')
    plt.scatter(x=codebook[:, 1], y=codebook[:, 0], c='r')
    plt.show()
    print(codebook)
    # # 使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    # label = vq(data, centroid)[0]


def every_x_minutes_visual(filepath, x=10):
    df = pd.read_csv(filepath)
    df = df[['Start_Time']].copy()
    print(df.head(10))
    date_arr = []
    num_arr = []

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    t1 = datetime.strptime('2016/3/22 19:36', '%Y/%m/%d %H:%M')
    t_end = datetime.strptime('2020/12/31 21:29', '%Y/%m/%d %H:%M')
    t2 = t1 + timedelta(minutes=x)
    while t2 < t_end:
        date_arr.append(t1)
        temp = df[t1 <= df.Start_Time]
        num_arr.append(temp[temp.Start_Time < t2].shape[0])
        t1 = t2
        t2 = t1 + timedelta(minutes=x)
    date_arr.append(t1)
    num_arr.append(df[df.Start_Time >= t1].shape[0])

    dff = pd.DataFrame({'Date_Start': date_arr, 'nums_{}min'.format(x): num_arr})
    print(dff.head(10))
    dff.to_csv(r'D:\毕业设计\github代码\1在看\preprocessed-datasets\nums_every10min.csv')


def every_3000m_visual(filepath):
    df = pd.read_csv(filepath)
    df = df[['grid_3000m', 'Year']].copy()
    # df2016 = df[df.Year == 2016].copy()
    # df2017 = df[df.Year == 2017].copy()
    # df2018 = df[df.Year == 2018].copy()
    # df2019 = df[df.Year == 2019].copy()
    # df2020 = df[df.Year == 2020].copy()
    # print(df.head(10))
    # grid_3000m = []
    # num_arr = []

    dff = df.groupby(['grid_3000m', 'Year']).size()
    print(dff['11S LT  21 23'])
    # print(dff.head(10))
    dff.to_csv(r'D:\毕业设计\数据集\nums_every3000m.csv')


if __name__ == "__main__":
    df = r'D:\毕业设计\数据集\datasets_2020_Los_final.csv'
    main(df)
