import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_utils import Spline2D

import numpy as np



def draw_shape(xx, yy, out_name, n_t=100):

    spline = Spline2D(xx,yy)

    x_s = np.zeros(n_t*len(xx))
    y_s = np.zeros(n_t*len(xx))

    t = np.arange(0, 1.0001, 1.0/(n_t-1))
    #assert len(t) == len(xx)
    for ii in range(len(xx)):
        for i_t in range(len(t)):
            pt = spline.values(ii, t[i_t])
            x_s[ii*len(t)+i_t] = pt[0]
            y_s[ii*len(t)+i_t] = pt[1]

    v0 = spline.values(0,0)
    v1 = spline.values(len(xx)-1,1)

    d0 = spline.derivatives(0,0)
    d1 = spline.derivatives(len(xx)-1,1)

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.plot(x_s, y_s, c='b', zorder=1)
    plt.scatter(xx, yy, c='r', zorder=2, alpha=0.4)
    _xlim = plt.xlim()
    _ylim = plt.ylim()

    def scale_x(x_v):
        return (x_v-_xlim[0])*1000/(_xlim[1]-_xlim[0])
    def scale_y(y_v):
        return (y_v-_ylim[0])*1000/(_ylim[1]-_ylim[0])


    plt.subplot(2,2,2)
    img = np.zeros((1000,1000), dtype=float)
    plt.imshow(img, zorder=1, cmap='gray')
    plt.plot(scale_x(x_s), scale_y(y_s), color='b', zorder=2)
    plt.scatter(scale_x(xx), scale_y(yy), color='r', zorder=3, alpha=0.5)

    plt.savefig(out_name)

if __name__ == "__main__":

    xx = np.array([2, 4, 5, 6, 9, 6, 5, 4])
    yy = np.array([4,5,8,5,4,3,1,3])
    draw_shape(xx, yy, 'star.pdf')

    xx = np.array([2,7,3,4,5,11,7,6])
    yy = np.array([3,6,9,10,10,4,4,1])
    assert len(xx) == len(yy)
    draw_shape(xx, yy, 'snake.pdf')

    xx = np.array([1,5,5,7,7,1])
    yy = np.array([1,1,7,7,0,0])
    draw_shape(xx,yy,'ell.pdf')
