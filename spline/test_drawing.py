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
    for ii in range(len(xx)):
        for i_t in range(len(t)):
            pt = spline.values(ii, t[i_t])
            x_s[ii*len(t)+i_t] = pt[0]
            y_s[ii*len(t)+i_t] = pt[1]

    resolution = 15.0/1000.0

    def scale_x(x_v):
        return x_v/resolution
    def scale_y(y_v):
        return y_v/resolution

    img = np.zeros((1000,1000), dtype=float)
    ann = Annotation(xx, yy)
    mask = ann.get_mask(resolution)

    dx = min(mask.shape[1], img.shape[1]-ann.x_min)
    dy = min(mask.shape[0], img.shape[0]-ann.y_min)

    print('before ',img.sum())

    ix0 = np.round(ann.x_min/resolution).astype(int)
    iy0 = np.round(ann.y_min/resolution).astype(int)

    img[iy0:iy0+dy, ix0:ix0+dx] = mask

    y0 = ann.y_min-50
    y1 = ann.y_min+dy+50
    x0 = ann.x_min-50
    x1 = ann.x_min+dx+50

    blank_mask = np.zeros(mask.shape, dtype=bool)

    blank_mask[ann.border_y_pixels, ann.border_x_pixels] = True

    img[iy0:iy0+dy, ix0:ix0+dx][blank_mask] = 2.0

    xy = np.array([xx,yy])
    pix = ann.wc_to_pixel(xy)


    print('after ',img.sum())
    plt.figure(figsize=(10,10))
    plt.imshow(img, zorder=1, cmap='gray')
    plt.plot(scale_x(x_s), scale_y(y_s), color='c', zorder=2, alpha=0.3, linewidth=1)
    plt.scatter(ix0+pix[0,:], iy0+pix[1,:], color='r', zorder=3, alpha=0.5)
    #plt.scatter(ann._cx+ann.x_min, ann._cy+ann.y_min, zorder=4, color='g')
    x0 = max(0, ix0+ann.border_x_pixels.min()-50)
    x1 = ix0+ann.border_x_pixels.max()+50
    y0 = max(iy0+ann.border_y_pixels.min()-50, 0)
    y1 = iy0+ann.border_y_pixels.max()+50

    plt.xlim((x0, x1))
    plt.ylim((y0, y1))
    plt.savefig(out_name)
    plt.close()

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

    #assert m5_control.hexdigest() == m5_test.hexdigest()

    #exit()
    #print(mask.sum())
    #print(ann._x_min,ann._y_min)

    #print(mask.shape[0]*mask.shape[1])
    #exit()


if __name__ == "__main__":

    xx = np.array([2, 4, 5, 6, 9, 5.8, 4.9, 4])+5
    yy = np.array([4,5,8,5,4,3,0.2,3])
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
