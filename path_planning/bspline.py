import matplotlib.pyplot as plt
from geomdl import BSpline
import numpy as np
import matplotlib.tri as tri


def bspline_curve2d():
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = [[5.0, 5.0], [10.0, 10.0], [20.0, 15.0], [35.0, 15.0], [45.0, 10.0], [50.0, 5.0]]
    # curve.knotvector = [0.0, 0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0, 1.0]
    curve.knotvector = [0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    curve.sample_size = 10
    curve_points = curve.evalpts

    for i in range(len(curve_points) - 1):
        plt.plot([curve_points[i][0], curve_points[i + 1][0]], [curve_points[i][1], curve_points[i + 1][1]], ls='-',
                 c='g', marker='.', mfc='g', mec='g')

    x_s = [1, 0, 3]
    y_s = [2, 8, 5]

    triangles = tri.Triangulation(x_s, y_s)
    plt.triplot(triangles, c='black', linestyle='-')
    plt.show()


if __name__ == "__main__":
    # main()
    bspline_curve2d()
