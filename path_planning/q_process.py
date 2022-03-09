import pandas as pd
import numpy as np


# # 自定义map函数,将Canvas下的坐标转换为直角坐标系下的坐标
# def test_map(x):
#     return [(x[0] - 3) / 20, (x[1] - 3) / 20]


def index_change(filepath):
    df = pd.read_csv(filepath)
    # df['x'] = df['coordinate'].apply(lambda x: (x.split('[')[1]).split(',')[0])
    # df['y'] = df['coordinate'].apply(lambda x: (x.split('[')[1]).split(',')[1])
    # df['x'] = df['x'].apply(lambda x: (int(x) - 3) / 20)
    # df['y'] = df['y'].apply(lambda x: 24 - ((int(x) - 3) / 20))
    df['xx'] = df['x'].apply(lambda x: int(x) + 0.5)
    df['yy'] = df['y'].apply(lambda x: int(x) + 0.5)
    df.to_csv(r'D:\毕业设计\数据集\q_table_map_final.csv')


def add_index(filepath):
    df = pd.read_csv(filepath)
    df['i'] = df['x'].map(str) + ',' + df['y'].map(str)
    df.to_csv(r'D:\毕业设计\数据集\q_table_map_final2.csv')


if __name__ == "__main__":
    df = r'D:\毕业设计\数据集\q_table_map_final.csv'
    # index_change(df)
    add_index(df)
