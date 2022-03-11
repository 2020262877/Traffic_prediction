import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import seaborn as sns
import tqdm
from matplotlib import ticker
from grid_process import LLtoUSNG
import time


def process(filepath_accident, filepath_census):
    df = pd.read_csv(filepath_accident)
    dff = pd.read_csv(filepath_census)
    print('US_Accidents_Dec20_updated.csv {} Rows {} Columns'.format(df.shape[0], df.shape[1]))
    print(df.head(3))
    print(df.describe())

    dff.info()
    dff = dff.drop(['Type', 'Counties', 'Latitude', 'Longitude'], axis=1)
    dff.info()

    # 查看数据列的缺省情况
    print(df.isna().mean().sort_values(ascending=False))

    a = (df.isna().sum().sort_values(ascending=False) / len(df)) * 100
    plt.title("Percentage of null values in the dataset", size=17, color="grey")
    plt.xlabel('\n Percentage (%) \n', fontsize=15, color='grey')
    plt.ylabel('\n Columns \n', fontsize=15, color='grey')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=12)
    a[a != 0].plot(kind="barh", color="pink")
    plt.show()

    # 删除缺省 70% 数据的 'Number'列
    df = df.drop('Number', axis=1)

    # 将 Start_Time 和 End_Time 转化为 datetime 类型
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    # 提取 year, month, day, hour and weekday
    df['Year'] = df['Start_Time'].dt.year
    df['Month'] = df['Start_Time'].dt.strftime('%b')
    df['Day'] = df['Start_Time'].dt.day
    df['Hour'] = df['Start_Time'].dt.hour
    df['Minute'] = df['Start_Time'].dt.minute
    df['Weekday'] = df['Start_Time'].dt.strftime('%a')

    # 事故持续时间（分钟）向上取整
    df['Time_Duration(min)'] = round((df['End_Time'] - df['Start_Time']) / np.timedelta64(1, 'm'))

    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020.csv')
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\processed_us2021census.csv')


def datamining_usa(filepath, filepath2):
    df = pd.read_csv(filepath)
    dff = pd.read_csv(filepath2)

    # # 每年的交通事故量条形图
    # df.groupby('Year').size().plot(kind='bar', align='center', color='coral')
    # plt.ylabel('Number of Records')
    # plt.show()
    #
    # 交通事故数量州排名
    # df.groupby('State').size().sort_values(ascending=False).head(10).plot(kind='bar', align='center', color='coral')
    # plt.title('Traffic Accidents of USA in (2016-2020)', color='grey')
    # plt.xlabel('State', color='grey')
    # plt.ylabel('Number of Records', color='grey')
    # plt.ylabel('Number of Records')
    # plt.show()
    #
    # # 交通事故数量城市排名
    # df.groupby('City').size().sort_values(ascending=False).head(10).plot(kind='bar', align='center', color='coral')
    # plt.title('Traffic Accidents of USA in (2016-2020)', color='grey')
    # plt.xlabel('City', color='grey')
    # plt.ylabel('Number of Records', color='grey')
    # plt.show()

    # 城市人口数量排名
    dff.groupby('City')['Population'].sum().sort_values(ascending=False).head(10).plot(kind='bar', align='center',
                                                                                       color='coral')
    plt.title('Population number of USA in (2016-2020)', color='grey')
    plt.xlabel('City', color='grey')
    plt.ylabel('Number of Records', color='grey')
    plt.show()
    # dff.groupby("City")["Population"].sum().sort_values(ascending=False).head(10)


def datamining_city(filepath):
    df = pd.read_csv(filepath)
    city_name = 'Los Angeles'
    # state_name = 'CA'

    # df_ca = df[df.State == state_name]
    df_los = df[df.City == city_name]

    # # 城市的交通事故量 (2016-2020)
    # plt.title('Traffic Accidents of {} in (2016-2020)'.format(city_name), color='grey')
    # plt.xlabel('Year', color='grey')
    # plt.ylabel('Number of Records', color='grey')
    # df_los.groupby('Year').size().plot(kind='bar', align='center', color='coral')
    # plt.show()

    # # 城市的交通事故量 (2016-2020) pivot 表格, 形成dataframe结果 有错！！！！！
    # plt.title('Traffic Accidents of {} in (2016-2020)'.format(city_name), color='grey')
    # plt.xlabel('Year', color='grey')
    # plt.ylabel('Months', color='grey')
    # sns.heatmap(df_los.pivot('Month', 'Year', df_los.groupby['Month', 'Year'].size()))
    # plt.show()

    # # 城市交通事故分布图
    # plt.title('Traffic Accident Severity Distribution of {} in 2020'.format(city_name), color='grey')
    # plt.xlabel('Longitude', color='grey')
    # plt.ylabel('Latitude', color='grey')
    # sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df_los[df_los.Year == 2020], hue='Severity', legend=False)
    # plt.show()

    # 城市 top 10 高风险道路
    street = pd.DataFrame(
        df_los.Street.value_counts().reset_index().rename(
            columns={'index': 'Street No.', 'Street': 'Cases'}))
    top_street = pd.DataFrame(street.head(10))
    plt.title('Top 10 Accident Prone Streets in {} (2016-2020)'.format(city_name), color='grey')
    plt.xlabel('Street No.', color='grey')
    plt.ylabel('Accident Cases', color='grey')
    plt.xticks(rotation=270)
    a = sns.barplot(x=top_street['Street No.'], y=top_street.Cases)  # palette="rainbow"
    a.yaxis.set_major_formatter(ticker.EngFormatter())
    # plt.show()
    print(len(street.Cases))

    xx = 0
    for (i, j) in zip(top_street["Street No."], range(0, 10)):
        xx += top_street.Cases[j] / len(df.Street) * 100
        print("Percentage of accident cases on street: {} is {:.2f}%".format(i, (
                top_street.Cases[j] / len(df.Street) * 100)))
    print('Top-10 percentage of accident cases is {:.2f}%'.format(xx))


def month_analysis(filepath):
    df = pd.read_csv(filepath)
    # city_name = 'Houston'
    df = df[df.State == 'CA']
    # df = df[df.City == city_name]

    df = df[df.Year > 2016]
    df = df[df.Year < 2020]
    plt.title('Accident cases in each month of {} in (2017-2019)'.format('California'), color='grey')
    plt.xlabel('Month', color='grey')
    plt.ylabel('Accident Cases', color='grey')
    sns.histplot(df.sort_values(by='month_123').Month, bins=7, kde=False)
    plt.show()

    # dff = df[df.Year == 2019]
    # plt.title('Accident cases in each month of {} in 2019'.format('Los Angeles'), color='grey')
    # plt.xlabel('Month', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(dff.sort_values(by='month_123').Month, bins=7, kde=False)
    # plt.show()

    # dff = df[df.Year == 2017]
    # plt.title('Accident cases in each month of {} in 2017'.format('Los Angeles'), color='grey')
    # plt.xlabel('Month', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(dff.sort_values(by='month_123').Month, bins=7, kde=False)
    # plt.show()


def week_analysis(filepath):
    df = pd.read_csv(filepath)
    city_name = 'Los Angeles'
    # 城市 一周
    # plt.title('Accident cases in each week of {}'.format(city_name), color='grey')
    # plt.xlabel('Weekday', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(df.sort_values(by='weekday_123').Weekday, bins=7, kde=False, color='darkorange')
    # plt.show()
    #
    # plt.title('Accident cases in each week of {}'.format(city_name), color='grey')
    # plt.xlabel('Weekday', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.violinplot(x='Weekday', y='Hour', hue='Sunrise_Sunset', split='False', data=df.sort_values(by='weekday_123'))
    # plt.show()

    plt.title('Accident cases in each week of {}'.format(city_name), color='grey')
    plt.xlabel('Weekday', color='grey')
    plt.ylabel('Accident Cases', color='grey')
    sns.violinplot(x='Weekday', y='Hour', data=df.sort_values(by='weekday_123'))
    plt.show()

    # # 城市周一与周末事故发生对比
    # plt.title('Accident cases occurs on Monday of {}'.format(city_name), color='grey')
    # plt.xlabel('Hour', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(df[df.Weekday == 'Mon'].Hour, bins=7, kde=False, color='orangered')
    # plt.show()
    #
    # plt.title('Accident cases occurs on Saturday of {}'.format(city_name), color='grey')
    # plt.xlabel('Hour', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(df[df.Weekday == 'Sat'].Hour, bins=7, kde=False, color='b')
    # plt.show()


def day_analysis(filepath):
    df = pd.read_csv(filepath)
    # df = df[df.City == 'Los Angeles']
    city_name = 'Los Angeles'
    # # 一天
    # plt.title('Accident cases for a day of {}'.format(city_name), color='grey')
    # plt.xlabel('Hour', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(df.Hour, bins=24, kde=False, color='lightcoral')
    # plt.show()

    # 周六周日
    dff = df[df.weekday_123 > 5]
    plt.title('Accident cases for a day of {} in weekend'.format(city_name), color='grey')
    plt.xlabel('Hour', color='grey')
    plt.ylabel('Accident Cases', color='grey')
    # plt.yticks([0, 1, 2, 3, 4, 5], [0, 500, 1000, 1500, 2000, 2500])
    sns.histplot(dff.Hour, bins=24, kde=False, color='dodgerblue')
    plt.ylim(0, 1000)
    plt.xticks([0, 4.79, 10.538, 16.286, 23], ['0:00', '6:00', '12:00', '18:00', '24:00'])
    plt.yticks([0, 200, 400, 600, 800, 1000], [0, 100, 200, 300, 400, 500])

    plt.show()

    # # 周内
    # dff = df[df.weekday_123 < 6]
    # plt.title('Accident cases for a day of {} in working day'.format(city_name), color='grey')
    # plt.xlabel('Hour', color='grey')
    # plt.ylabel('Accident Cases', color='grey')
    # sns.histplot(dff.Hour, bins=24, kde=False, color='khaki')
    # plt.yticks([0, 500, 1000, 1500, 2000, 2500], [0, 100, 200, 300, 400, 500])
    # plt.xticks([0, 4.79, 10.538, 16.286, 23], ['0:00', '6:00', '12:00', '18:00', '24:00'])
    # plt.show()


def temp_h_mining(filepath):
    # city_name = 'Los Angeles'
    state_name = 'California'
    # state_name = 'TX'
    df = pd.read_csv(filepath)
    df = df[df.State == 'CA']
    # df = df[df.City == city_name]

    # temp = pd.DataFrame(df['Temperature(F)'].value_counts()).reset_index().rename(
    #     columns={'index': 'Temp', 'Temperature(F)': 'Cases'})
    # plt.title('Traffic Accidents in different temperature ranges of {}'.format(city_name), color='grey')
    # plt.xlabel('Accident Numbers', color='grey')
    # plt.ylabel('Temperature(F)', color='grey')
    # sns.scatterplot(x=temp.Cases[temp.Cases > 1], y=temp.Temp, color='r')  # [(t - 32) / 1.8 for t in temp.Temp]
    # plt.show()

    # humi = pd.DataFrame(df['Humidity(%)'].value_counts()).reset_index().rename(
    #     columns={'index': 'Humidity', 'Humidity(%)': 'Cases'})
    # plt.title('Traffic Accidents in different Humidity of {}'.format(state_name), color='grey')
    # plt.xlabel('Accident Numbers', color='grey')
    # plt.ylabel('Humidity(%)', color='grey')
    # sns.scatterplot(x=humi.Cases, y=humi.Humidity, color='royalblue')
    # plt.show()

    # x = np.array(humi.Cases).reshape(-1, 1)
    # y = np.array(humi.Humidity).reshape(-1, 1)

    # # 线性回归
    # model_ln = lm.LinearRegression()
    # model_ln.fit(x, y)
    # pred_y_ln = model_ln.predict(x)

    # # 岭回归
    # model_rd = lm.Ridge(150, fit_intercept=True, max_iter=1000)
    # model_rd.fit(x, y)
    # pred_y_rd = model_rd.predict(x)
    #
    # sorted_indices = x.T[0].argsort()
    # plt.plot(x[sorted_indices], pred_y_ln[sorted_indices], c='orangered', label='Linear')
    # plt.plot(x[sorted_indices], pred_y_rd[sorted_indices], c='limegreen', label='Ridge')
    # plt.show()

    df = df[df.Year > 2016]
    df = df[df.Year < 2020]
    # df = df[df.Year == 2019]
    mons_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mons_nums = [0] * 12
    mons_temp = [0] * 12
    mons_humi = [0] * 12

    sel_col = ['Month', 'Temperature(F)', 'Humidity(%)', 'Weather_Condition']
    df_sel = df[sel_col].copy()
    # #  丢弃有缺失值的行
    # df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    # df_sel.shape

    temp_group = df_sel.groupby('Month')['Temperature(F)'].sum()
    humi_group = df_sel.groupby('Month')['Humidity(%)'].sum()
    mons_group = df_sel.groupby('Month').size()

    for i in range(12):
        temps = temp_group[mons_name[i]]
        humis = humi_group[mons_name[i]]
        mons_nums[i] = mons_group[mons_name[i]]
        mons_temp[i] = temps / mons_nums[i]
        mons_humi[i] = humis / mons_nums[i]

    # 每月湿度变化
    plt.subplot2grid((3, 3), (0, 0), colspan=4)
    plt.title('Temperature and Humidity in each month of {}'.format(state_name), color='grey')
    plt.ylabel('Temperature (F)')
    plt.plot(mons_name, mons_temp, ls='-', c='coral', marker='.', mfc='coral', mec='coral')

    plt.subplot2grid((3, 3), (1, 0), colspan=4)
    # plt.title('Humidity in each month of {}'.format(city_name), color='grey')
    plt.ylabel('Humidity (%)')
    plt.plot(mons_name, mons_humi, ls='-', c='c', marker='.', mfc='c', mec='c')

    plt.subplot2grid((3, 3), (2, 0), colspan=4)
    # plt.title('Traffic Accidents in each month of {}'.format(city_name), color='grey')
    plt.xlabel('Month', color='grey')
    plt.ylabel('Traffic Accidents')
    plt.plot(mons_name, mons_nums, ls='-', c='dimgray', marker='.', mfc='dimgray', mec='dimgray')
    plt.show()


def road_mining(filepath):
    city_name = 'Los Angeles'
    df = pd.read_csv(filepath)
    df = df[df.City == city_name]

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))

    road_conditions = ['Bump', 'Crossing', 'Give_Way', 'Junction', 'Stop', 'No_Exit', 'Traffic_Signal', 'Turning_Loop']
    colors = [('#6662b3', '#00FF00'), ('#7881ff', '#0e1ce8'), ('#18f2c7', '#09ad8c'), ('#08ff83', '#02a352'),
              ('#ffcf87', '#f5ab3d'),
              ('#f5f53d', '#949410'), ('#ff9187', '#ffc7c2'), ('tomato', '#008000')]
    count = 0

    def func(pct, allvals):
        absolute = int(round(pct / 100 * np.sum(allvals), 2))
        return "{:.2f}%\n({:,d} Cases)".format(pct, absolute)

    for i in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
        size = list(df[road_conditions[count]].value_counts())
        if len(size) != 2:
            size.append(0)
        labels = ['False', 'True']
        i.pie(size, labels=labels, colors=colors[count],
              autopct=lambda pct: func(pct, size), labeldistance=1.1, explode=[0, 0.2], textprops={'fontsize': 10})
        title = 'Presence of {}'.format(road_conditions[count])
        i.set_title(title, color='grey')
        count += 1

    plt.show()


def process_grid(filepath, precision=3000):
    city_name = 'Los Angeles'
    df = pd.read_csv(filepath)
    df = df[df.City == city_name]
    df.info()
    print(df.head(3))

    df['grid_1000m'] = [LLtoUSNG(lat, lng, 1000) for lat, lng in zip(df.Start_Lat, df.Start_Lng)]
    # df['grid_2000m'] = [LLtoUSNG(lat, lng, precision) for lat, lng in zip(df.Start_Lat, df.Start_Lng)]
    # df['grid_3000m'] = [LLtoUSNG(lat, lng, 3000) for lat, lng in zip(df.Start_Lat, df.Start_Lng)]
    # df_ca = df[df.State == 'CA']
    df_los = df[df.City == 'Los Angeles']

    # df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_grids.csv')
    # df_ca.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_CA_grids.csv')
    df_los.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_grids2.csv')


def street_visual(filepath):
    # 待续
    df = pd.read_csv(filepath)
    df.info()
    print(df.head(3))
    df_street_gp = df.groupby('Street').size().sort_values(ascending=False)
    #
    # for street in tqdm.tqdm(set(df.Street)):
    #     p = df[df.Street == street].Severity.tolist()
    #     p1 = [0] + p[:-1]
    #     p2 = [0, 0] + p[:-2]
    #     p3 = [0, 0, 0] + p[:-3]
    #     df.loc[df[df.Street == street].index, 'p1_street'] = p1[:len(p)]
    #     df.loc[df[df.Street == street].index, 'p2_street'] = p2[:len(p)]
    #     df.loc[df[df.Street == street].index, 'p3_street'] = p3[:len(p)]
    # df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_grids3_street.csv')


def grid_visual(filepath):
    df = pd.read_csv(filepath)
    for grid_idx in tqdm.tqdm(set(df.grid_3000m)):
        p = df[df.grid_3000m == grid_idx].Severity.tolist()
        p1 = [0] + p[:-1]
        p2 = [0, 0] + p[:-2]
        p3 = [0, 0, 0] + p[:-3]
        df.loc[df[df.grid_3000m == grid_idx].index, 'p1_3000m'] = p1[:len(p)]
        df.loc[df[df.grid_3000m == grid_idx].index, 'p2_3000m'] = p2[:len(p)]
        df.loc[df[df.grid_3000m == grid_idx].index, 'p3_3000m'] = p3[:len(p)]
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_grids3.csv')


def timestamp_add(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = [time.mktime(time.strptime(t, '%Y/%m/%d %H:%M')) for t in df.Start_Time]

    df = df.sort_values(by='timestamp')
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_grids3_timestamp.csv')


def add_seasons(filepath):
    df = pd.read_csv(filepath)
    seasons = ['seasons'] * len(df)
    for x in range(0, len(df)):
        m = df.loc[x, 'Month']
        if m in ['Dec', 'Jan', 'Feb']:
            seasons[x] = 'Winter'
        elif m in ['Mar', 'Apr', 'May']:
            seasons[x] = 'Spring'
        elif m in ['Jun', 'Jul', 'Aug']:
            seasons[x] = 'Summer'
        elif m in ['Sep', 'Oct', 'Nov']:
            seasons[x] = 'Fall'
    df['seasons'] = seasons
    df.info()
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_final.csv')


def add_grid_no(filepath):
    grids = []
    df = pd.read_csv(filepath)
    grid_no = [0] * len(df)
    for x in range(0, len(df)):
        m = df.loc[x, 'grid_3000m']
        if m not in grids:
            grids.append(m)
        grid_no[x] = grids.index(m) + 1
    df['grid_3k_no'] = grid_no
    print(df.head(3))
    print(len(grids))
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_final.csv')


def add_street_flag(filepath):
    streets_info = {}
    df = pd.read_csv(filepath)
    street_flag = []

    for x in range(0, len(df)):
        m = df.loc[x, 'Street']
        if m not in streets_info:
            streets_info[m] = 1
        else:
            streets_info[m] = streets_info[m] + 1

    for x in range(0, len(df)):
        m = df.loc[x, 'Street']
        num = streets_info[m]
        if num <= 5:
            street_flag.append(1)
        elif num <= 10:
            street_flag.append(2)
        elif num <= 39:
            street_flag.append(3)
        elif num <= 100:
            street_flag.append(4)
        elif num <= 300:
            street_flag.append(5)
        elif num <= 500:
            street_flag.append(6)
        elif num <= 700:
            street_flag.append(7)
        elif num <= 1000:
            street_flag.append(8)
        elif num <= 2000:
            street_flag.append(9)
        else:
            street_flag.append(10)

    df['street_flag'] = street_flag
    print(df.head(3))
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_Los_final2.csv')


def add_month_weekday_123(filepath):
    df = pd.read_csv(filepath)
    df['weekday_123'] = df['Weekday'].map({'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7})
    df['month_123'] = df['Month'].map(
        {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
         'Nov': 11, 'Dec': 12})
    df.to_csv(r'D:\毕业设计\数据集\preprocessed-datasets\datasets_2020_final.csv')


def weather_mining(filepath):
    df = pd.read_csv(filepath)
    for x in ["Fog", "Light Rain", "Rain", "Heavy Rain", "Smoke"]:
        plt.subplots(1, 2, figsize=(10, 6))
        plt.subplot(1, 2, 1)
        df.loc[df["Weather_Condition"] == x]["Severity"].value_counts().sort_values().plot(kind="bar")
        plt.suptitle("Severity for " + str(x), fontsize=20)
        plt.xlabel("severity")
        plt.ylabel("number_of_deaths")
        plt.subplot(1, 2, 2)
        df.loc[df["Weather_Condition"] == x]['Severity'].value_counts().plot.pie(autopct='%1.0f%%', fontsize=16)
    plt.show()


if __name__ == "__main__":
    df = r'D:\毕业设计\数据集\datasets_2020_Los_final.csv'
    day_analysis(df)
    # df = pd.read_csv(df)
