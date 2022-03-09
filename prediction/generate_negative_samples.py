from random import random
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import shap


def generate(filepath):
    cols = ['Start_Lng', 'Start_Lat', 'Side', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'target',
            'Visibility(mi)', 'Wind_Direction', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
            'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal',
            'Turning_Loop', 'Timezone', 'Sunrise_Sunset', 'Year', 'month_123', 'weekday_123', 'Hour', 'Minute',
            'grid_3k_no', 'street_flag']
    # 读取数据
    # mnist = fetch_openml('MNIST original', data_home=filepath)
    # 得到数据X和标签y
    df = pd.read_csv(filepath)
    df = df[cols]

    df.isnull().mean()
    #  丢弃有缺失值的行
    df.dropna(subset=df.columns[df.isnull().mean() != 0], how='any', axis=0, inplace=True)
    df.shape
    df.info()

    df = pd.get_dummies(df, drop_first=True)
    y = df['target']
    X = df.drop('target', axis=1)
    # 标签的副本拷贝
    y_orig = y.copy()

    select_index = np.array(np.where(y == 1)).reshape(-1).tolist()
    # 随机选择size大小的index作为positive数据
    select_index_size = np.random.choice(select_index, replace=False, size=1000)
    # 其他的标签都改成0，认为是unlabelled
    other_index = [i for i in range(len(X)) if i not in select_index_size]
    y[other_index] = -1
    print(y)

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rf.fit(X, y)

    results = pd.DataFrame({
        'truth': y_orig,  # True labels
        'label': y,  # Labels shown to models
        'output_std': rf.predict_proba(X)[:, 1]  # Random forest's scores
    }, columns=['truth', 'label', 'output_std'])

    # 利用可靠负样例与确定的正样例重新训练个分类器，再对无标签数据进行再次预测
    # 拿到预测结果
    prob = rf.predict_proba(X)[:, 1]
    # 预测到的可靠负样本为38714个
    select_all_index_negative = np.array(np.where(prob == 0)).reshape(-1).tolist()
    # 按正负样本1:3随机抽取
    select_index_negative = np.random.choice(select_all_index_negative, replace=False, size=10000)
    # 记录剩余的未标注的数据的索引
    other_index_unlabel = [i for i in range(len(X)) if i not in select_index_size and i not in select_index_negative]
    # 将可靠负样例标记为0
    y[select_index_negative] = 0

    # 正样例的索引为 select_index_size，标签为1， 负样例的索引为select_index_negative，标签为0，
    # 待测的数据索引集为select_index_negative，为-1，构造训练集
    X_postive = X[select_index_size]
    X_negative = X[select_index_negative]
    y_postive = y[select_index_size]
    y_negative = y[select_index_negative]
    X_train = np.concatenate((X_postive, X_negative), axis=0)
    y_train = np.concatenate((y_postive, y_negative), axis=0)
    X_test = X[other_index_unlabel]

    rf1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
    rf1.fit(X_train, y_train)

    predicts = rf1.predict_proba(X_test)[:, 1]
    predicts.shape

    results1 = pd.DataFrame({
        'truth': y_orig[other_index_unlabel],  # True labels
        'label': y[other_index_unlabel],  # Labels shown to models
        'output_std': predicts  # Random forest's scores
    }, columns=['truth', 'label', 'output_std'])
    results1[(results1['output_std'] > 0.9) & (results1['truth'] == 1)]

    # 模型输出的正样本为A，真正的正样本集合为B
    # 计算精确率（Precision）,指的是模型判别为正的所有样本中有多少是真正的正样本,则Precision（A,B）=|A∩B| / |A|
    A_B = len(results1[(results1['output_std'] >= 0.08) & (results1['truth'] == 1)])
    A = len(results1[results1['output_std'] >= 0.08])
    Precision_A_B = A_B / A * 100
    print("直接应用标准分类器，计算精确率Precision = %.3f %%" % Precision_A_B)
    # 计算召回率（Recall）,指的是所有正样本有多少被模型判定为正样本Recall（A,B） = |A∩B| / |B|
    B = len(results1[results1['truth'] == 1])
    Recall_A_B = A_B / B * 100
    print("直接应用标准分类器，计算精确率Recall = %.3f %%" % Recall_A_B)


if __name__ == "__main__":
    path = r'D:\毕业设计\数据集\test\datasets_2020_Los_final.csv'
    generate(path)
