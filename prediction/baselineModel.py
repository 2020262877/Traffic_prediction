from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # plotting

import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
shap.initjs()
import PUAdapter


def outliers_removing_test(df):
    # df.info()
    # Q1:  40.0     Q2:  93.0     Q3:  360.0    Max:  10153    Min: 8   normal range:(0, 840.0)
    # Print time_duration information
    dff = df['Time_Duration']
    sns.boxplot(dff)
    s = pd.Series(dff)
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.5)
    q3 = s.quantile(0.75)
    print('Mean: {}\tQ1: {}\tQ2: {}\tQ3: {}\tMax: {}\tMin:{}\t'.format(s.median(), q1, q2, q3, s.max(), s.min()))

    # 离群点删除
    iqr = q3 - q1
    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr
    print('normal range:({}, {})'.format(lower, upper))
    df = df[df.Time_Duration < upper]
    df.info()


def baseline(df):
    df = df[df.Year == 2020]
    # df.info()
    # Q1:  40.0     Q2:  93.0     Q3:  360.0    Max:  10153    Min: 8   normal range:(0, 840.0)
    # Print time_duration information
    dff = df['Time_Duration']
    sns.boxplot(dff)
    s = pd.Series(dff)
    q1 = s.quantile(0.25)
    q2 = s.quantile(0.5)
    q3 = s.quantile(0.75)
    print('Mean: {}\tQ1: {}\tQ2: {}\tQ3: {}\tMax: {}\tMin:{}\t'.format(s.median(), q1, q2, q3, s.max(), s.min()))

    # 离群点删除
    iqr = q3 - q1
    # lower = min           (0, q1 - 1.5 * iqr)
    lower = 0
    upper = q3 + 1.5 * iqr
    print('normal range:({}, {})'.format(lower, upper))
    # Set the list of features to include in Machine Learning

    df = df.loc[df.Time_Duration < upper].copy()
    df.info()

    feature_lst = ['Severity', 'Start_Lng', 'Start_Lat', 'Side', 'Temperature(F)',
                   'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Weather_Condition', 'Amenity',
                   'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                   'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Timezone', 'Sunrise_Sunset', 'Year',
                   'month_123', 'weekday_123', 'Hour', 'grid_3k_no', 'street_flag']

    df_sel = df[feature_lst].copy()

    # Check missing values 空 -> true -> 1
    df_sel.isnull().mean()

    #  丢弃有缺失值的行
    df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean() != 0], how='any', axis=0, inplace=True)
    df_sel.shape
    df_sel.info()

    # 为分类数据生成哑元dummies (类似独热编码)
    df_train = pd.get_dummies(df_sel, drop_first=True)
    df_train.info()

    target = 'Severity'

    y = df_train[target]
    X = df_train.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

    # lr = LogisticRegression(random_state=0)  # , max_iter=300
    # lr.fit(X_train, y_train)  # 训练
    # y_pred = lr.predict(X_test)  # 测试
    # acc = accuracy_score(y_test, y_pred)
    #
    # print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
    # print(classification_report(y_test, y_pred))
    #
    # output(y_pred, df_test)

    algo_lst = ['Logistic Regression', ' K-Nearest Neighbors', 'Decision Trees', 'Random Forest', 'Deep Neural NetWork']
    accuracy_lst = []
    train_time_lst = []
    test_time_lst = []

    # ------------------------------------------SVC Test--------------------------------

    # ---------------------------------LogisticRegression classifier------------------------------------------------
    lr = LogisticRegression(random_state=0)  # , max_iter=300
    t1 = datetime.now()
    lr.fit(X_train, y_train)  # 训练
    t2 = datetime.now()
    y_pred = lr.predict(X_test)  # 测试

    train_time_lst.append(t2 - t1)
    test_time_lst.append(datetime.now() - t2)
    acc = accuracy_score(y_test, y_pred)
    accuracy_lst.append(acc)

    print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
    print("train time: {}".format(train_time_lst[-1]))
    print("test time: {}".format(test_time_lst[-1]))
    print(classification_report(y_test, y_pred))

    # ---------------------------------k-NN classifier------------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=6)
    t1 = datetime.now()
    knn.fit(X_train, y_train)
    t2 = datetime.now()
    y_pred = knn.predict(X_test)

    train_time_lst.append(t2 - t1)
    test_time_lst.append(datetime.now() - t2)
    acc = accuracy_score(y_test, y_pred)
    accuracy_lst.append(acc)

    # print('[K-Nearest Neighbors (KNN)] knn.score: {:.3f}.'.format(knn.score(X_test, y_test)))
    print('[K-Nearest Neighbors (KNN)] accuracy_score: {:.3f}.'.format(acc))
    print("train time: {}".format(train_time_lst[-1]))
    print("test time: {}".format(test_time_lst[-1]))
    print(classification_report(y_test, y_pred))

    # ---------------------------------DecisionTree classifier---------------------------------------------------
    dt = DecisionTreeClassifier()
    t1 = datetime.now()
    dt.fit(X_train, y_train)
    t2 = datetime.now()
    y_pred = dt.predict(X_test)

    train_time_lst.append(t2 - t1)
    test_time_lst.append(datetime.now() - t2)
    acc = accuracy_score(y_test, y_pred)
    accuracy_lst.append(acc)

    # print('[Decision Trees (DT)] knn.score: {:.3f}.'.format(dt.score(X_test, y_test)))
    print("[Decision Trees] accuracy_score: {:.3f}.".format(acc))
    print("train time: {}".format(train_time_lst[-1]))
    print("test time: {}".format(test_time_lst[-1]))
    print(classification_report(y_test, y_pred))

    # explainer = shap.TreeExplainer(dt)
    # # top = X_test.sample(n=200)
    # shap_values = explainer.shap_values(X_test)
    # # shap.summary_plot(shap_values, X_test, plot_type='bar')
    # # plt.show()
    # shap.force_plot(explainer.expected_value[0], shap_values[0])
    # plt.show()

    # ---------------------------------RandomForest classifier---------------------------------------------------
    rf = RandomForestClassifier()
    t1 = datetime.now()
    rf.fit(X_train, y_train)
    t2 = datetime.now()
    y_pred = rf.predict(X_test)
    print(y_pred)

    train_time_lst.append(t2 - t1)
    test_time_lst.append(datetime.now() - t2)
    acc = accuracy_score(y_test, y_pred)
    accuracy_lst.append(acc)

    # print('[Random Forest (RF)] knn.score: {:.3f}.'.format(rf.score(X_test, y_test)))
    print('[Random Forest (RF)] accuracy_score: {:.3f}.'.format(acc))
    print("train time: {}".format(train_time_lst[-1]))
    print("test time: {}".format(test_time_lst[-1]))
    print(classification_report(y_test, y_pred))

    # ---------------------------------MLPC classifier (NN)------------------------------------------------
    dnn = MLPClassifier(hidden_layer_sizes=(50, 50), alpha=0.01, max_iter=300, solver='sgd', activation='relu')
    t1 = datetime.now()
    dnn.fit(X_train, y_train)
    t2 = datetime.now()
    y_pred = dnn.predict(X_test)

    train_time_lst.append(t2 - t1)
    test_time_lst.append(datetime.now() - t2)
    acc = accuracy_score(y_test, y_pred)
    accuracy_lst.append(acc)

    print('[Deep Neural Network (DNN)] accuracy_score: {:.3f}.'.format(acc))
    print("train time: {}".format(train_time_lst[-1]))
    print("test time: {}".format(test_time_lst[-1]))
    print(classification_report(y_test, y_pred))

    for i in range(len(algo_lst)):
        print(algo_lst[i] + '---' + 'accuracy:{:.3f}    训练时间:{}     测试时间:{}'.format(accuracy_lst[i], train_time_lst[i],
                                                                                    test_time_lst[i]))


def output(y_pred, X_test):
    X_test.info()
    y = y_pred[:10]
    x = X_test.head(10)
    out = {}
    for i in range(0, len(x)):
        m = x.loc[i, 'grid_3k_no']
        s = y[i]
        t = str(x.loc[i, 'Year']) + '/' + str(x.loc[i, 'month_123']) + '/' + str(x.loc[i, 'Day']) + ' ' + str(
            x.loc[i, 'Hour']) + 'h'
        if m not in out.keys():
            out[m] = [[s], [t]]
        else:
            val = out.get(m)
            val[0].append(s)
            val[1].append(t)
            out[m] = val
    print(out)


def rec(m, n, tol):
    if type(m) != 'numpy.ndarray':
        m = np.array(m)
    if type(n) != 'numpy.ndarray':
        n = np.array(n)
    l = m.size
    percent = 0
    for i in range(l):
        if np.abs(10 ** m[i] - 10 ** n[i]) <= tol:
            percent += 1
    return 100 * (percent / l)


if __name__ == "__main__":
    df = pd.read_csv(r'D:\毕业设计\数据集\datasets_2020_Los_final.csv')
    baseline(df)
