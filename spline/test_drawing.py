import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spline_utils import Spline2D, Annotation

import numpy as np

import os
import hashlib


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

    resolution = 15.0/1000.0
    plt.xlim(0,15)
    plt.ylim(0,15)

    def scale_x(x_v):
        return x_v/resolution
    def scale_y(y_v):
        return y_v/resolution

    plt.subplot(2,2,2)
    img = np.zeros((1000,1000), dtype=float)
    plt.imshow(img, zorder=1, cmap='gray')
    plt.plot(scale_x(x_s), scale_y(y_s), color='b', zorder=2)
    plt.scatter(scale_x(xx), scale_y(yy), color='r', zorder=3, alpha=0.5)

    plt.subplot(2,2,3)
    ann = Annotation(xx, yy, resolution)
    plt.scatter(ann._border_x_pixels_by_x, ann._border_y_pixels_by_x)

    cx = -ann._x_min+ann._spline.x.sum()/(len(ann._spline.x)*resolution)
    cy = -ann._y_min+ann._spline.y.sum()/(len(ann._spline.y)*resolution)

    #cx, cy = ann.get_mask()

    plt.scatter([cx], [cy], color='r')
    plt.scatter(-ann._x_min+ann._spline.x/ann.resolution,
                -ann._y_min+ann._spline.y/ann.resolution,
                c='c')

    mask = ann.get_mask()

    dx = min(mask.shape[1], img.shape[1]-ann._x_min)
    dy = min(mask.shape[0], img.shape[0]-ann._y_min)

    print('before ',img.sum())

    img[ann._y_min:ann._y_min+dy,
        ann._x_min:ann._x_min+dx] = mask
    print('after ',img.sum())
    plt.subplot(2,2,4)
    plt.imshow(img, zorder=1, cmap='gray')
    plt.plot(scale_x(x_s), scale_y(y_s), color='c', zorder=2, alpha=0.2)
    plt.scatter(scale_x(xx), scale_y(yy), color='r', zorder=3, alpha=0.5)
    plt.scatter(ann._cx+ann._x_min, ann._cy+ann._y_min, zorder=4, color='g')
    print('saving %s' % out_name)
    print(mask.sum())
    plt.savefig(out_name)

    m5_control = hashlib.md5()
    with open(os.path.join('test_figs/%s' % os.path.basename(out_name)), 'rb') as in_file:
        while True:
            data = in_file.read(10000)
            if len(data)==0:
                break
            m5_control.update(data)

    m5_test = hashlib.md5()
    with open(out_name, 'rb') as in_file:
        while True:
            data = in_file.read(10000)
            if len(data)==0:
                break
            m5_test.update(data)

    assert m5_control.hexdigest() == m5_test.hexdigest()

    #exit()
    #print(mask.sum())
    #print(ann._x_min,ann._y_min)

    #print(mask.shape[0]*mask.shape[1])
    #exit()


if __name__ == "__main__":

    xx = np.array([2, 4, 5, 6, 9, 6, 5, 4])+5
    yy = np.array([4,5,8,5,4,3,1,3])
    draw_shape(xx, yy, 'star.png')

    xx = np.array([2,7,3,4,5,11,7,6])
    yy = np.array([3,6,9,10,10,4,4,1])
    assert len(xx) == len(yy)
    draw_shape(xx, yy, 'snake.png')

    xx = np.array([1,5,5,7,7,1])+5
    yy = np.array([1,1,7,7,0,0])+5
    draw_shape(xx,yy,'ell.png')

    xx = np.array([1,3,6,7,8,9,7,5,4,3])
    yy = np.array([2,6,6,2,4,3,1,3,4,2])
    draw_shape(xx,yy,'tilde.png')

    pts = [(2,2),(1,5),(3,7),(6,7),(8,6),
           (7,2), (6,3), (7,5), (6,6),
           (4,6), (3,4), (3,3)]
    xx = np.zeros(len(pts), dtype=float)
    yy = np.zeros(len(pts), dtype=float)
    for ii in range(len(pts)):
        xx[ii] =pts[ii][0]
        yy[ii] = pts[ii][1]
    draw_shape(xx,yy,'horseshoe.png')
