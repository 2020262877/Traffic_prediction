import math
import matplotlib.pyplot as plt
from random import uniform
from math import sqrt, atan2, cos, sin, tan, pi
from datetime import datetime
import matplotlib.tri as tri
import pandas as pd
import numpy as np
from geomdl import BSpline

filepath = r'D:\毕业设计\数据集\q_table_map_final.csv'
df = pd.read_csv(filepath)
df = df.drop('i', axis=1)
df.set_index(['x', 'y'])


def main():
    bounds = [
        (0, 0),
        (25, 0),
        (25, 25),
        (0, 25)
    ]

    ws = Workspace(bounds)
    ws.plot()

    start = State(1, 1)
    end = State(20, 20)

    # r、y、c、b 分别代表上下左右
    obs_state = DynamicObstacleState(8, 10, 'black', 1)
    obs_state2 = DynamicObstacleState(14, 15, 'black', 1)

    uav = UAV(start, end)
    d_obs = DynamicObstacle(obs_state)
    d_obs.ss.append(obs_state2)

    # 生成5个不同形状的静态障碍
    s_obs_state1 = StaticObstacleState(8, 7, 'cir', 1.5)
    s_obs_state2 = StaticObstacleState(13, 11, 'rect', 5, 1)
    s_obs_state3 = StaticObstacleState(18, 20, 'rect', 1, 3)
    s_obs_state4 = StaticObstacleState(9, 20, 'rect', 5, 1)
    s_obs_state5 = StaticObstacleState(19, 10, 'rect', 1, 5)
    s_obs_state6 = StaticObstacleState(20, 18, 'rect', 5, 1)
    s_obs_state7 = StaticObstacleState(20, 22, 'rect', 5, 1)
    s_obs_state8 = StaticObstacleState(11, 17, 'rect', 1, 5)
    s_obs = StaticObstacle(s_obs_state1)
    s_obs.ss.append(s_obs_state2)
    s_obs.ss.append(s_obs_state3)
    s_obs.ss.append(s_obs_state4)
    s_obs.ss.append(s_obs_state5)
    s_obs.ss.append(s_obs_state6)
    s_obs.ss.append(s_obs_state7)
    s_obs.ss.append(s_obs_state8)

    start_time = datetime.now()
    for i in range(0, 5000):
        if tree_expansion(uav, d_obs, s_obs, 1, 0.4):
            print('sampling points number: ' + str(i))
            break

    time = datetime.now() - start_time
    cost = uav.tree[-1][0].cost
    print('cost (distance): ' + str(cost))
    print('path planning time: ' + str(time))
    print('microseconds:{}'.format(time.microseconds))
    # return cost, time.microseconds
    d_obs.plot()
    s_obs.plot()
    uav.plot_tree()
    uav.plot()
    uav.plot_goal()
    plt.axis('equal')
    plt.show()


def tree_expansion(uav, obs, s_obs, step=1, threshold=0.4):
    # 重新界定采样空间
    n_last = uav.tree[-1][0]

    # 对从扩展树最后一个节点到目的地的路径进行单位步长采样
    angle = n_last.return_angle(uav.end)
    x_s = n_last.x + step * cos(angle)
    y_s = n_last.y + step * sin(angle)
    # 采样节点 x_s
    n_new = State(x_s, y_s, cost=uav.tree[-1][0].cost + step)

    # 判断是否满足约束
    cons = Constrains(obs.ss, s_obs.ss, n_new)
    if cons.scope_constrain():
        uav.tree.append((n_new, n_last))
    else:
        rep_state = State(math.floor(n_last.x), math.floor(n_last.y))
        uav.tree[-1] = (rep_state, uav.tree[-1][1])
        x = rep_state.x + 0.5
        y = rep_state.y + 0.5
        while not cons.scope_constrain():
            state = df.loc[(df['xx'] == x) & (df['yy'] == y)]
            state = state.iloc[0]
            state = state.drop(['x', 'y', 'xx', 'yy'])
            state = state.reindex(np.random.permutation(state.index))
            # print('random_state:{}'.format(state))
            action = state.idxmax()
            if action == '0':
                xx, yy = x, y + 1
            elif action == '1':
                xx, yy = x, y - 1
            elif action == '2':
                xx, yy = x + 1, y
            else:
                xx, yy = x - 1, y
            n_new = State(xx + 0.5, yy + 0.5, 0)
            uav.tree.append((n_new, uav.tree[-1][0]))
            x = xx
            y = yy
            cons.s = n_new

        uav.tree.append((uav.end, uav.tree[-1][0]))
    return n_new.return_distance(uav.end) <= 1


class State:
    def __init__(self, x, y, cost=0):
        self.x = x
        self.y = y
        self.cost = cost

    def plot(self, color='k', m='o'):
        plt.plot(self.x, self.y, c=color, marker=m)

    def return_distance(self, node):
        return sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def return_angle(self, node):
        return atan2(node.y - self.y, node.x - self.x)


class Workspace:
    def __init__(self, bounds, obstacles=None):
        self.bounds = bounds
        self.obstacles = obstacles

    def plot(self):
        x_ords = []
        y_ords = []
        for coordinate in self.bounds:
            x_ords.append(coordinate[0])
            y_ords.append(coordinate[1])

        x_ords.append(x_ords[0])
        y_ords.append(y_ords[0])

        plt.plot(x_ords, y_ords, color='black')

    def return_sample_space(self):
        x_min = self.return_min_bound(0)
        x_max = self.return_max_bound(0)
        y_min = self.return_min_bound(1)
        y_max = self.return_max_bound(1)

        return [
            (x_min, x_max),
            (y_min, y_max)
        ]

    def return_min_bound(self, index):
        min_val = self.bounds[0][index]

        for coordinate in self.bounds:
            if coordinate[index] < min_val:
                min_val = coordinate[index]

        return min_val

    def return_max_bound(self, index):
        max_val = self.bounds[0][index]

        for coordinate in self.bounds:
            if coordinate[index] > max_val:
                max_val = coordinate[index]

        return max_val


class UAV:
    def __init__(self, start, end, color='red'):
        self.start = start
        self.root = start
        self.end = end
        self.tree = [(start, start)]
        self.color = color

    def plot(self):
        self.root.plot(self.color, m='o')

    def plot_goal(self):
        self.end.plot(self.color, m='^')

    def plot_tree(self, color='green'):
        points = [[self.tree[0][1].x, self.tree[0][1].y]]
        xx = []
        yy = []
        for node in self.tree:
            points.append([node[0].x, node[1].y])
            x_ords = [node[1].x, node[0].x]
            y_ords = [node[1].y, node[0].y]
            xx.append(node[0].x)
            yy.append(node[0].y)
            plt.plot(x_ords, y_ords, ls='-', c=color, marker='.', mfc=color, mec=color)
        # print('x[]:{}'.format(xx))
        # print('y[]:{}'.format(yy))
        # bspline_curve2d(points, len(self.tree), 2)


class DynamicObstacleState:
    def __init__(self, x, y, direct, speed, r=2):
        """
        动态障碍（无人机）的状态
        :param x: 障碍 UAV 的坐标 (x, y)
        :param y:
        :param direct: 障碍 UAV 方向
        :param speed: 障碍 UAV 速度 每个 step 移动多少
        :param r: 障碍 UAV 的半径
        """
        self.x = x
        self.y = y
        self.direct = direct
        self.speed = speed
        self.r = r

    def plot(self, m='X'):
        plt.plot(self.x, self.y, c=self.direct, marker=m)


class DynamicObstacle:
    def __init__(self, s, r=2):
        self.ss = [s]
        self.r = r

    def plot(self):
        for s in self.ss:
            s.plot()


# 静态障碍 （建筑物）
class StaticObstacleState:
    def __init__(self, x, y, form, *parm):
        """
        :param x: 障碍物的中心坐标 (x, y)
        :param y:
        :param form: 障碍物的形状 ['rect', 'cir', 'tri']
        :param parm: 用于描述障碍物的参数 ['rect': parm=[矩形长, 矩形宽], 'cir': parm=[圆半径], 'tri': parm=[三角形边长]]
        """
        self.x = x
        self.y = y
        self.form = form
        self.parm = parm

    def plot(self, c='black', fill_color='grey', alpha=0.3):
        if self.form == 'rect':
            l = self.parm[0]
            w = self.parm[1]
            bounds = [
                (self.x - l / 2, self.y - w / 2),
                (self.x + l / 2, self.y - w / 2),
                (self.x + l / 2, self.y + w / 2),
                (self.x - l / 2, self.y + w / 2)
            ]
            x_s = []
            y_s = []
            for coordinate in bounds:
                x_s.append(coordinate[0])
                y_s.append(coordinate[1])

            x_s.append(x_s[0])
            y_s.append(y_s[0])
            # 打印矩形
            plt.plot(x_s, y_s, color=c)
            # 填充矩形
            plt.fill(x_s, y_s, color=fill_color, alpha=alpha)
        elif self.form == 'cir':
            r = self.parm[0]
            x_s = []
            y_s = []
            xx = self.x
            yy = self.y
            for i in range(0, 365):
                a = math.radians(i)
                x_s.append(xx + r * cos(a))
                y_s.append(yy + r * sin(a))
            plt.fill(x_s, y_s, color=fill_color, alpha=alpha)
            plt.plot(x_s, y_s, color=c)
        else:
            x = self.x
            y = self.y
            l = self.parm[0]
            x_s = [x, x - l / 2, x + l / 2]
            y_s = [y + l / 2 / cos(pi / 6), y - tan(pi / 6) * l / 2, y - tan(pi / 6) * l / 2]

            triangles = tri.Triangulation(x_s, y_s)
            plt.triplot(triangles, c='black', linestyle='-')
            plt.fill(x_s, y_s, color=fill_color, alpha=alpha)


class StaticObstacle:
    def __init__(self, s):
        self.ss = [s]

    def plot(self):
        for s in self.ss:
            s.plot()


class Constrains:
    def __init__(self, dyn_obs_s_set, sta_obs_s_set, s, step=1, threshold=0.4):
        self.dyn_obs_s_set = dyn_obs_s_set
        self.sta_obs_s_set = sta_obs_s_set
        self.s = s
        self.step = step
        self.threshold = threshold

    def scope_constrain(self):
        # 扫描静态障碍，判断是否满足约束
        for sta_obs_s in self.sta_obs_s_set:
            x = sta_obs_s.x
            y = sta_obs_s.y
            # print(sta_obs_s.form + ': ({:},{:})'.format(x, y))
            if sta_obs_s.form == 'cir':
                r = sta_obs_s.parm[0]
                is_qualify = sqrt((x - self.s.x) ** 2 + (y - self.s.y) ** 2) > (r + self.threshold)
                if not is_qualify:
                    return False
            elif sta_obs_s.form == 'rect':
                l = sta_obs_s.parm[0]
                w = sta_obs_s.parm[1]
                is_qualify = (self.s.x <= (x - l / 2) - self.threshold) | (self.s.x >= (x + l / 2) + self.threshold) | (
                        self.s.y <= (y - w / 2) - self.threshold) | (self.s.y >= (y + w / 2) + self.threshold)
                if not is_qualify:
                    return False
            else:
                r = (sta_obs_s.parm[0] / 2) / cos(pi / 6)
                is_qualify = sqrt((x - self.s.x) ** 2 + (y - self.s.y) ** 2) > (r + self.threshold)
                if not is_qualify:
                    return False

        # 扫描动态障碍，判断是否满足约束
        for dyn_obs_s in self.dyn_obs_s_set:
            x = dyn_obs_s.x
            y = dyn_obs_s.y
            is_qualify = sqrt((x - self.s.x) ** 2 + (y - self.s.y) ** 2) > 0.9
            if not is_qualify:
                return False

        return True


def bspline_curve2d(points, size, degree):
    curve = BSpline.Curve()
    curve.degree = degree
    curve.ctrlpts = points
    # curve.knotvector = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]
    knot_vector = []
    points_len = len(points)
    for i in range(degree):
        knot_vector.append(0.0)
    for i in range(points_len - degree):
        knot_vector.append(round(1 / (points_len - degree) * i, 2))
    for i in range(degree + 1):
        knot_vector.append(1.0)
    print(knot_vector)
    print(points_len)
    print(len(knot_vector))
    curve.knotvector = knot_vector
    curve.sample_size = size
    curve_points = curve.evalpts

    for i in range(len(curve_points) - 1):
        plt.plot([curve_points[i][0], curve_points[i + 1][0]], [curve_points[i][1], curve_points[i + 1][1]], ls='-',
                 c='y', marker=',', mfc='y', mec='y')
    # plt.show()


if __name__ == "__main__":
    # sum_time = 0
    # sum_cost = 0
    main()
    # df.info()
    # for i in range(10):
    #     cost, time = main()
    #     sum_cost += cost
    #     sum_time += time
    # print('Avg_cost:{}'.format(sum_cost / 10))
    # print('Avg_time(milliseconds):{}'.format(sum_time / 10 / 1000))
