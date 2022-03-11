import numpy
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import numpy as np  # linear algebra
from scipy.stats import spearmanr, pearsonr
import pmdarima as pm
from matplotlib import pyplot as plt
from scipy import stats
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.preprocessing import StandardScaler
# import os  # accessing directory structure
# import altair as alt
import tqdm


def fun3(df):
    # sns.set(palette="colorblind", color_codes=True, font='SimHei', font_scale=0.8)
    sns.set_context('paper')
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决无法显示符号的问题

    counts = df[df.State == 'CA'].groupby('grid_1000m').size().sort_values(ascending=False)

    ax = sns.boxplot(data=counts[(counts > 5) & (counts <= 20)])
    plt.show()

    # select the cell with max number of accidents from CA
    df[df.State == 'CA'].groupby('grid_1000m').size().sort_values(ascending=False)

    # Accodents per day of the grid cell with maximum number of accident records
    freq = df[df.grid_1000m == '11S LT  98 56'].sort_values(by='Start_Time') \
        .groupby(by=[pd.to_datetime(df.Start_Time) \
                 .map(lambda x: (x.year, x.month, x.day))]) \
        .size() \
        .sort_values(ascending=False)
    print(freq)
    sns.displot(freq, kde=True)
    plt.show()


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


def fun1(df):
    temp_df = df[df.grid_1000m == '11S LT  98 56'].sort_values(by='Start_Time').Severity.to_frame().reset_index(
        drop=True).values

    data, labels = univariate_data(temp_df, 0, len(temp_df) - 1, 5, 1)

    print(temp_df.shape)
    print(data.shape)
    print(labels.shape)

    for i in range(5):
        corr = spearmanr(data[:, i, :].flatten(), labels.flatten())
        print(corr)


def fun2(df):
    df.info()
    for state in ['CA', 'TX', 'SC', 'FL', 'NC']:
        df_state = df[df.State == state]
        for grid_idx in tqdm.tqdm(set(df_state.grid_1000m)):
            var = df_state[df_state.grid_1000m == grid_idx].Severity.tolist()
            var_1 = [0] + var[:-1]
            var_2 = [0, 0] + var[:-2]
            var_3 = [0, 0, 0] + var[:-3]
            df_state.loc[df_state[df_state.grid_1000m == grid_idx].index, 'previous_1'] = var_1[:len(var)]
            df_state.loc[df_state[df_state.grid_1000m == grid_idx].index, 'previous_2'] = var_2[:len(var)]
            df_state.loc[df_state[df_state.grid_1000m == grid_idx].index, 'previous_3'] = var_3[:len(var)]
        df_state.to_csv(f'output/accident_data_{state}.csv')


def environment_correlation(df):
    mons_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mons_nums = [0] * 12
    mons_temp = [0] * 12
    mons_humi = [0] * 12

    weather_nums = []

    sel_col = ['Month', 'Temperature(F)', 'Humidity(%)', 'Weather_Condition']
    df_sel = df[sel_col].copy()
    #  丢弃有缺失值的行
    df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    df_sel.shape

    temp_group = df_sel.groupby('Month')['Temperature(F)'].sum()
    humi_group = df_sel.groupby('Month')['Humidity(%)'].sum()
    mons_group = df_sel.groupby('Month').size()
    weather_nums = df_sel.groupby('Weather_Condition').size()
    print(weather_nums)

    for i in range(12):
        temps = temp_group[mons_name[i]]
        humis = humi_group[mons_name[i]]
        mons_nums[i] = mons_group[mons_name[i]]
        mons_temp[i] = (temps / mons_nums[i] - 32) / 1.8
        mons_humi[i] = humis / mons_nums[i]

    plt.subplot2grid((3, 3), (0, 0), colspan=4)
    plt.ylabel('Temperature (℃)')
    plt.plot(mons_name, mons_temp, ls='-', c='coral', marker='.', mfc='coral', mec='coral')

    plt.subplot2grid((3, 3), (1, 0), colspan=4)
    plt.ylabel('Humidity (%)')
    plt.plot(mons_name, mons_humi, ls='-', c='c', marker='.', mfc='c', mec='c')

    plt.subplot2grid((3, 3), (2, 0), colspan=4)
    plt.ylabel('Traffic Accidents')
    plt.plot(mons_name, mons_nums, ls='-', c='dimgray', marker='.', mfc='dimgray', mec='dimgray')
    plt.show()


def time_correlation(df):
    df.info()
    # mons_nums = [0] * 12
    hour_nums = [0] * 24
    hour_arr = []
    weekday_numeric = []

    sel_col = ['Severity', 'Month', 'Weekday', 'Day', 'Hour']
    df_sel = df[sel_col].copy()
    #  丢弃有缺失值的行
    df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    df_sel.shape

    for i in range(len(df_sel['Weekday'])):
        w = df_sel['Weekday'][i]
        if w == 'Mon':
            weekday_numeric.append(1)
        elif w == 'Tue':
            weekday_numeric.append(2)
        elif w == 'Wed':
            weekday_numeric.append(3)
        elif w == 'Thu':
            weekday_numeric.append(4)
        elif w == 'Fri':
            weekday_numeric.append(5)
        elif w == 'Sat':
            weekday_numeric.append(6)
        elif w == 'Sun':
            weekday_numeric.append(7)

    df_sel['Weekday_numeric'] = weekday_numeric
    hour_col = df['Hour'].copy()

    for i in hour_col:
        hour_arr.append(i)

    hour_group = df_sel.groupby('Hour').size()

    for i in range(24):
        hour_nums[i] = hour_group[i]

    # print(df_sel['Severity'][0])

    sns.displot(hour_arr, kde=True)
    plt.xlabel('time (h)')
    plt.show()

    sns.violinplot(x='Weekday', y='Hour', hue='Sunrise_Sunset', split='False',
                   data=df[['Weekday', 'Hour', 'Sunrise_Sunset']].copy())
    plt.show()

    sns.violinplot(x='Month', y='Weekday_numeric', data=df_sel)
    plt.show()

    sns.boxplot(x='Month', y='Weekday_numeric', data=df_sel)
    plt.show()

    # sns.stripplot(x="Weekday", y="Hour", data=df_sel, jitter=True)
    # plt.show()


def spatial_correlation(df):
    sign_num = []
    sign_num_y = []

    sel_col = ['Severity', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway',
               'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    df_sel = df[sel_col].copy()

    #  丢弃有缺失值的行
    df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    df_sel.shape

    for i in range(len(df_sel['Severity'])):
        n = 0
        for sign in sel_col[1:]:
            if df_sel[sign][i]:
                n += 1
        sign_num.append(n)

    df_sel['Sign_Num'] = sign_num
    print(df_sel['Sign_Num'])

    sign_group = df_sel.groupby('Sign_Num').size()
    print(sign_group)

    for s in sign_group:
        sign_num_y.append(s)

    # plt.plot([0, 1, 2, 3, 4, 5], sign_num_y, ls='-', c='r', marker='.', mfc='r', mec='r')
    sns.pointplot(x=[0, 1, 2, 3, 4, 5], y=sign_num_y)
    plt.xlabel('Sign Numbers')
    plt.ylabel('Traffic Accidents')
    plt.show()


if __name__ == "__main__":
    # df = pd.read_csv(r'D:\毕业设计\数据集\accident_data_CA.csv')
    # time_correlation(df)
    # environment_correlation(df)
    # spatial_correlation(df)

    # week_name = ['', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    # xx = numpy.where(df['Weekday'] == 'Mon')
    # tips = sns.load_dataset('tips')
    # tips.info()
    for i in range(100000):
        for j in range(100000):
            i * j / 3
