import matplotlib.pyplot as plt
from random import uniform
from math import sqrt, atan2, cos, sin
from datetime import datetime
from geomdl import BSpline
import numpy as np


def main():
    start_time = datetime.now()
    bounds = [
        (0, 0),
        (25, 0),
        (25, 25),
        (0, 25)
    ]

    ws = Workspace(bounds, None, bounds)
    ws.plot()

    start = State(5, 5)
    end = State(20, 20)
    # r、y、c、b 分别代表上下左右
    obs_state = ObstacleState(8, 9, 'grey', 1)
    obs_state2 = ObstacleState(14, 14, 'grey', 1)

    uav = UAV(start, end)
    obs = Obstacle(obs_state)
    obs.ss.append(obs_state2)

    # 在样本空间内随机生成 20 个障碍点
    for i in range(20):
        obs.ss.append(ObstacleState(uniform(5, 20), uniform(5, 20), 'grey', 1))

    for i in range(0, 10000):
        if tree_expansion(ws, uav, obs, 1, 1):
            print('sampling points number: ' + str(i))
            break

    print('cost (distance): ' + str(uav.tree[-1][0].cost))
    print('path planning time: ' + str(datetime.now() - start_time))
    obs.plot()
    uav.plot_tree()
    uav.plot()
    uav.plot_goal()
    plt.axis('equal')

    # x = y = np.arange(-4, 4, 0.1)
    # x, y = np.meshgrid(x, y)
    # plt.contour(x, y, (x - 10) ** 2 + (y - 10) ** 2, [2], colors='black')  # x**2 + y**2 = 9 的圆形

    plt.show()


def tree_expansion(ws, uav, obs, step=1, threshold=0.2):
    # 扩展树最后一个节点
    n_last = uav.tree[-1][0]
    angle = n_last.return_angle(uav.end)
    x_s = n_last.x + step * cos(angle)
    y_s = n_last.y + step * sin(angle)
    is_liner = True

    # 根据扩展树最后一个采样节点，更改采样空间
    # new_bounds = [
    #     (n_last.x, n_last.y),
    #     (25, n_last.y),
    #     (25, 25),
    #     (n_last.x, 25)
    # ]
    ws.cur_bounds[0] = (n_last.x, n_last.y)
    ws.cur_bounds[1] = (ws.cur_bounds[1][0], n_last.y)
    ws.cur_bounds[3] = (n_last.x, ws.cur_bounds[1][1])
    s_s = ws.return_sample_space()

    # 生成采样节点 x_s
    x_s = State(x_s, y_s)

    # 判断是否满足约束
    is_qualify, n_near, x_s = Constrains(obs.ss, x_s, uav, step, is_liner).scope_constrain()
    while not is_qualify:
        x_s = uniform(s_s[0][0], s_s[0][1])
        y_s = uniform(s_s[1][0], s_s[1][1])
        x_s = State(x_s, y_s)
        is_liner = False
        is_qualify, n_near, x_s = Constrains(obs.ss, x_s, uav, step, is_liner).scope_constrain()

    uav.tree.append((x_s, n_near))
    return x_s.return_distance(uav.end) <= threshold


class State:
    def __init__(self, x, y, cost=0):
        self.x = x
        self.y = y
        self.cost = cost

    def plot(self, color='k', m='o'):
        plt.plot(self.x, self.y, marker=m, color=color)

    def return_distance(self, node):
        return sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def return_angle(self, node):
        return atan2(node.y - self.y, node.x - self.x)


class Workspace:
    def __init__(self, bounds, obstacles=None, cur_bounds=None):
        self.bounds = bounds
        self.obstacles = obstacles
        self.cur_bounds = cur_bounds

    def plot(self):
        x_ords = []
        y_ords = []
        for coordinate in self.bounds:
            x_ords.append(coordinate[0])
            y_ords.append(coordinate[1])

        x_ords.append(x_ords[0])
        y_ords.append(y_ords[0])

        plt.plot(x_ords, y_ords, color='orange')

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
        min_val = self.cur_bounds[0][index]

        for coordinate in self.cur_bounds:
            if coordinate[index] < min_val:
                min_val = coordinate[index]

        return min_val

    def return_max_bound(self, index):
        max_val = self.cur_bounds[0][index]

        for coordinate in self.cur_bounds:
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
        for node in self.tree:
            points.append([node[0].x, node[1].y])
            x_ords = [node[1].x, node[0].x]
            y_ords = [node[1].y, node[0].y]
            plt.plot(x_ords, y_ords, ls='-', c=color, marker=',', mfc=color, mec=color)
        bspline_curve2d(points, len(self.tree) * 3, 2)


class ObstacleState:
    def __init__(self, x, y, direct, speed, r=1):
        self.x = x
        self.y = y
        self.direct = direct
        self.speed = speed
        self.r = r

    def plot(self, m='X'):
        plt.plot(self.x, self.y, c=self.direct, marker=m)


class Obstacle:
    def __init__(self, s, r=2):
        self.ss = [s]
        self.r = r

    def plot(self):
        for s in self.ss:
            s.plot()


class Constrains:
    def __init__(self, obs_s_set, s, uav, step, is_liner):
        self.obs_s_set = obs_s_set
        self.s = s
        self.uav = uav
        self.step = step
        self.is_liner = is_liner

    def scope_constrain(self):
        is_qualify = True
        if self.is_liner:
            self.s.cost = self.uav.tree[-1][0].cost + self.step
            for obs_s in self.obs_s_set:
                is_qualify = is_qualify & (sqrt((obs_s.x - self.s.x) ** 2 + (obs_s.y - self.s.y) ** 2) > obs_s.r)
        else:
            # 归一化采样节点的坐标
            # 1.扩展树中与采样节点 x_s 距离最近的节点 n_near
            n_near = self.uav.tree[-1][0]
            min_dist = n_near.return_distance(self.s)

            # for node, _ in self.uav.tree:
            #     if node.return_distance(self.s) < min_dist:
            #         n_near = node
            #         min_dist = n_near.return_distance(self.s)

            # 2.将采样节点作为叶子节点连接到 n_near
            if min_dist < self.step:
                self.s.cost = min_dist + n_near.cost
            else:
                angle = n_near.return_angle(self.s)
                self.s.x = n_near.x + self.step * cos(angle)
                self.s.y = n_near.y + self.step * sin(angle)
                self.s.cost = n_near.cost + self.step

            for obs_s in self.obs_s_set:
                is_qualify = is_qualify & (sqrt((obs_s.x - self.s.x) ** 2 + (obs_s.y - self.s.y) ** 2) > obs_s.r)
        return is_qualify, self.uav.tree[-1][0], self.s


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
    main()
