import sys

import numpy
import pyfits

def weighted_quadrupole(im, centroid, sigma):
    y, x = numpy.indices(im.shape)
    gaussian = lambda y1, x1: numpy.exp(-0.5 * ((y1 - centroid[0])**2 \
                                                + (x1 - centroid[1])**2)/sigma**2)
    weighted_image = im * gaussian(y, x)
    total = weighted_image.sum()
    x2 = (weighted_image*(x-centroid[1])**2).sum()/total
    y2 = (weighted_image*(y-centroid[0])**2).sum()/total
    xy = (weighted_image*(x-centroid[1])*(y-centroid[0])).sum()/total
    # print centroid[::-1]
    # import matplotlib.pyplot as plt
    # plt.imshow(im)
    # plt.scatter([centroid[1]], [centroid[0]], marker="+", s=500, color="white")
    # plt.show()
    # plt.imshow(gaussian(y, x))
    # plt.scatter([centroid[1]], [centroid[0]], marker="+", s=500, color="white")
    # plt.show()
    return x2, y2, xy

def quadrupole(im, centroid, plot=False):
    y, x = numpy.indices(im.shape)
    total = im.sum()
    x2 = (im*(x-centroid[1])**2).sum()/total
    y2 = (im*(y-centroid[0])**2).sum()/total
    xy = (im*(x-centroid[1])*(y-centroid[0])).sum()/total

    # if plot:
    #     plt.subplot(2,2,1)
    #     plt.imshow(im, interpolation="nearest")
    #     plt.subplot(2,2,2)
    #     plt.imshow(im*(x-centroid[1])**2, interpolation="nearest")
    #     plt.colorbar()
    #     plt.subplot(2,1,2)
    #     hist = (im*(x-centroid[1])**2).sum(axis=0)
    #     bins = numpy.arange(im.shape[1])
    #     center = bins+0.5
    #     plt.bar(center, hist, align="center")
    #     plt.show()

    return x2, y2, xy

im = pyfits.getdata(sys.argv[1])
inCat = pyfits.getdata(sys.argv[2], ext=2)
Vx10 = []
Vy10 = []
Vx20 = []
Vy20 = []
Vx40 = []
Vy40 = []
Vx80 = []
Vy80 = []
Vx120 = []
Vy120 = []
wVx10 = []
wVy10 = []
wVx20 = []
wVy20 = []
wVx40 = []
wVy40 = []
wVx80 = []
wVy80 = []
wVx120 = []
wVy120 = []
for c in inCat:
    x = c["XWIN_IMAGE"] - 1
    y = c["YWIN_IMAGE"] - 1
    xint = numpy.rint(x)
    yint = numpy.rint(y)

    gaussian_sigma = 1.0 / 2.35 * 5.0 # 1 arcsec FWHM -> pixels sigma

    #10 pixel wide box
    xlow = xint - 5
    ylow = yint - 5
    thumb = im[ylow:ylow+11, xlow:xlow+11]
    x2, y2, xy = quadrupole(thumb, (y-ylow, x-xlow))
    wx2, wy2, wxy = weighted_quadrupole(thumb, (y-ylow, x-xlow), gaussian_sigma)
    Vx10.append(x2*(0.2/3600)**2)
    Vy10.append(y2*(0.2/3600)**2)
    wVx10.append(wx2*(0.2/3600)**2)
    wVy10.append(wy2*(0.2/3600)**2)

    #20 pixel wide box
    xlow = xint - 10
    ylow = yint - 10
    thumb = im[ylow:ylow+21, xlow:xlow+21]
    x2, y2, xy = quadrupole(thumb, (y-ylow, x-xlow))
    wx2, wy2, wxy = weighted_quadrupole(thumb, (y-ylow, x-xlow), gaussian_sigma)
    Vx20.append(x2*(0.2/3600)**2)
    Vy20.append(y2*(0.2/3600)**2)
    wVx20.append(wx2*(0.2/3600)**2)
    wVy20.append(wy2*(0.2/3600)**2)

    #40 pixel wide box
    xlow = xint - 20
    ylow = yint - 20
    thumb = im[ylow:ylow+41, xlow:xlow+41]
    x2, y2, xy = quadrupole(thumb, (y-ylow, x-xlow))
    wx2, wy2, wxy = weighted_quadrupole(thumb, (y-ylow, x-xlow), gaussian_sigma)
    Vx40.append(x2*(0.2/3600)**2)
    Vy40.append(y2*(0.2/3600)**2)
    wVx40.append(wx2*(0.2/3600)**2)
    wVy40.append(wy2*(0.2/3600)**2)

    #80 pixel wide box
    xlow = xint - 40
    ylow = yint - 40
    thumb = im[ylow:ylow+81, xlow:xlow+81]
    x2, y2, xy = quadrupole(thumb, (y-ylow, x-xlow))
    wx2, wy2, wxy = weighted_quadrupole(thumb, (y-ylow, x-xlow), gaussian_sigma)
    Vx80.append(x2*(0.2/3600)**2)
    Vy80.append(y2*(0.2/3600)**2)
    wVx80.append(wx2*(0.2/3600)**2)
    wVy80.append(wy2*(0.2/3600)**2)

    #120 pixel wide box
    xlow = xint - 60
    ylow = yint - 60
    thumb = im[ylow:ylow+121, xlow:xlow+121]
    x2, y2, xy = quadrupole(thumb, (y-ylow, x-xlow))
    wx2, wy2, wxy = weighted_quadrupole(thumb, (y-ylow, x-xlow), gaussian_sigma)
    Vx120.append(x2*(0.2/3600)**2)
    Vy120.append(y2*(0.2/3600)**2)
    wVx120.append(wx2*(0.2/3600)**2)
    wVy120.append(wy2*(0.2/3600)**2)

cVx10 = pyfits.Column(name="VX10", format="1D", array=Vx10)
cVy10 = pyfits.Column(name="VY10", format="1D", array=Vy10)
cVx20 = pyfits.Column(name="VX20", format="1D", array=Vx20)
cVy20 = pyfits.Column(name="VY20", format="1D", array=Vy20)
cVx40 = pyfits.Column(name="VX40", format="1D", array=Vx40)
cVy40 = pyfits.Column(name="VY40", format="1D", array=Vy40)
cVx80 = pyfits.Column(name="VX80", format="1D", array=Vx80)
cVy80 = pyfits.Column(name="VY80", format="1D", array=Vy80)
cVx120 = pyfits.Column(name="VX120", format="1D", array=Vx120)
cVy120 = pyfits.Column(name="VY120", format="1D", array=Vy120)
wcVx10 = pyfits.Column(name="WVX10", format="1D", array=wVx10)
wcVy10 = pyfits.Column(name="WVY10", format="1D", array=wVy10)
wcVx20 = pyfits.Column(name="WVX20", format="1D", array=wVx20)
wcVy20 = pyfits.Column(name="WVY20", format="1D", array=wVy20)
wcVx40 = pyfits.Column(name="WVX40", format="1D", array=wVx40)
wcVy40 = pyfits.Column(name="WVY40", format="1D", array=wVy40)
wcVx80 = pyfits.Column(name="WVX80", format="1D", array=wVx80)
wcVy80 = pyfits.Column(name="WVY80", format="1D", array=wVy80)
wcVx120 = pyfits.Column(name="WVX120", format="1D", array=wVx120)
wcVy120 = pyfits.Column(name="WVY120", format="1D", array=wVy120)
newHDU = pyfits.new_table([cVx10, cVy10,
                           cVx20, cVy20,
                           cVx40, cVy40,
                           cVx80, cVy80,
                           cVx120, cVy120,
                           wcVx10, wcVy10,
                           wcVx20, wcVy20,
                           wcVx40, wcVy40,
                           wcVx80, wcVy80,
                           wcVx120, wcVy120])
outCat = inCat.columns + newHDU.columns
outHDU = pyfits.new_table(outCat)
outFilename = sys.argv[2].replace(".fits", "_V.fits")
outHDU.writeto(outFilename)
