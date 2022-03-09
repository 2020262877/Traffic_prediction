import math
import matplotlib.pyplot as plt
from random import uniform
from math import sqrt, atan2, cos, sin, tan, pi
from datetime import datetime
import matplotlib.tri as tri


def main():
    start_time = datetime.now()
    bounds = [
        (0, 0),
        (25, 0),
        (25, 25),
        (0, 25)
    ]

    ws = Workspace(bounds)
    ws.plot()

    start = State(5, 5)
    end = State(20, 20)
    # r、y、c、b 分别代表上下左右
    obs_state = ObstacleState(8, 10, 'black', 1)
    obs_state2 = ObstacleState(14, 14, 'black', 1)

    uav = UAV(start, end)
    obs = Obstacle(obs_state)
    obs.ss.append(obs_state2)

    # 生成5个不同形状的静态障碍
    s_obs_state1 = StaticObstacleState(8, 7, 'cir', 1)
    s_obs_state2 = StaticObstacleState(13, 11, 'rect', 4, 2)
    s_obs_state3 = StaticObstacleState(18, 17, 'rect', 2, 2)
    s_obs_state4 = StaticObstacleState(10, 20, 'tri', 3)
    s_obs_state5 = StaticObstacleState(19, 10, 'rect', 1, 5)
    s_obs = StaticObstacle(s_obs_state1)
    s_obs.ss.append(s_obs_state2)
    s_obs.ss.append(s_obs_state3)
    s_obs.ss.append(s_obs_state4)
    s_obs.ss.append(s_obs_state5)

    for i in range(0, 10000):
        if tree_expansion(ws, uav, obs, s_obs, 1, 0.4):
            print('sampling points number: ' + str(i))
            break

    print('cost (distance): ' + str(uav.tree[-1][0].cost))
    print('path planning time: ' + str(datetime.now() - start_time))
    obs.plot()
    s_obs.plot()
    uav.plot_tree()
    uav.plot()
    uav.plot_goal()
    plt.axis('equal')
    plt.show()


def tree_expansion(ws, uav, obs, s_obs, step=1, threshold=0.4):
    s_s = ws.return_sample_space()

    # 范围内随机生成一个实数
    x_s = uniform(s_s[0][0], s_s[0][1])
    y_s = uniform(s_s[1][0], s_s[1][1])

    # 采样节点 x_s
    x_s = State(x_s, y_s, cost=0)

    # 寻找当前扩展树到采样节点距离的最短的节点
    n_near = uav.tree[0][0]
    min_dist = n_near.return_distance(x_s)

    for node, _ in uav.tree:
        if node.return_distance(x_s) < min_dist:
            n_near = node
            min_dist = n_near.return_distance(x_s)

    # expand towards nearest node
    if min_dist < step:
        n_new = x_s
        n_new.cost = n_near.cost + min_dist
    else:
        angle = n_near.return_angle(x_s)
        x_new = n_near.x + step * cos(angle)
        y_new = n_near.y + step * sin(angle)
        n_new = State(x_new, y_new, n_near.cost + step)

    # 判断是否满足约束
    cons = Constrains(obs.ss, s_obs.ss, n_new, uav, n_near)
    while not cons.scope_constrain():
        x_s = uniform(s_s[0][0], s_s[0][1])
        y_s = uniform(s_s[1][0], s_s[1][1])
        x_s = State(x_s, y_s, cost=0)

        n_near = uav.tree[0][0]
        min_dist = n_near.return_distance(x_s)
        for node, _ in uav.tree:
            if node.return_distance(x_s) < min_dist:
                n_near = node
                min_dist = n_near.return_distance(x_s)

        if min_dist < step:
            n_new = x_s
            n_new.cost = n_near.cost + min_dist
        else:
            angle = n_near.return_angle(x_s)
            x_new = n_near.x + step * cos(angle)
            y_new = n_near.y + step * sin(angle)
            n_new = State(x_new, y_new, n_near.cost + step)
        cons.s = n_new
        cons.n_near = n_near
    uav.tree.append((n_new, n_near))
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
        for node in self.tree:
            x_ords = [node[1].x, node[0].x]
            y_ords = [node[1].y, node[0].y]
            plt.plot(x_ords, y_ords, ls='-', c=color, marker='.', mfc=color, mec=color)


# 动态障碍（无人机）
class ObstacleState:
    def __init__(self, x, y, direct, speed, r=2):
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
    def __init__(self, obs_s_set, sta_obs_s_set, s, n_near, step=1, threshold=0.4):
        self.obs_s_set = obs_s_set
        self.sta_obs_s_set = sta_obs_s_set
        self.s = s
        self.n_near = n_near
        self.step = step
        self.threshold = threshold

    def scope_constrain(self):
        # 扫描静态障碍，判断是否满足约束
        for sta_obs_s in self.sta_obs_s_set:
            x = sta_obs_s.x
            y = sta_obs_s.y
            print(sta_obs_s.form + ': ({:},{:})'.format(x, y))
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
        return True


if __name__ == "__main__":
    main()
