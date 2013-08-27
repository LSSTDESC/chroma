import sys
import os
import gzip
import glob

import numpy

import astropy.utils.console
import matplotlib.pyplot as plt
import matplotlib as mpl

import multiprocessing as mp

def parallactic_zenith_angles(HA, dec):
    # latitude = -(30.0 + 14.0/60 + 26.7/3600) * numpy.pi / 180.0
    # coslat = numpy.cos(-latitude)
    # sinlat = numpy.sin(-latitude)
    # hard coding the above in for speed
    latitude = -0.5278006557724753
    coslat = 0.86391682470652653
    sinlat = -0.50363451032369966

    if isinstance(HA, int) or isinstance(HA, float):
        HA = numpy.array([HA])
        dec = numpy.array([dec])
    elif isinstance(HA, list) or isinstance(HA, tuple):
        HA = numpy.array(HA)
        dec = numpy.array(dec)
    elif not isinstance(HA, numpy.ndarray):
        raise TypeError

    cosdec = numpy.cos(dec)
    sindec = numpy.sin(dec)
    cosHA = numpy.cos(HA)
    sinHA = numpy.sin(HA)

    cosza = cosHA * cosdec * coslat + sindec * sinlat
    za = numpy.arccos(cosza) # za in [0, pi].  check
    sinza = numpy.sin(za)
    sinq = -sinHA * coslat / sinza
    cosq = (-sinlat + sindec * cosza) / (cosdec * sinza)
    q = numpy.arctan2(sinq, cosq)

    return za, q

def opsim_fields():
    f = gzip.open('output_opsim3_61.dat.gz','r')
    titles = f.readline().split()
    vals = f.readline().split()
    ii = range(len(titles))
    for i, title, val in zip(ii, titles, vals):
        print '{:02d} {:25} {}'.format(i, title, val)

def opsim_parse():
    n = 10000000 # only need about 3e6 of these in the end
    out = numpy.empty(n, dtype=[('fieldID', 'i4'),
                                ('filter', 'a1'),
                                ('airmass', 'f8'),
                                ('z_a', 'f8'),
                                ('z_a2', 'f8'),
                                ('HA', 'f8'),
                                ('q', 'f8'),
                                ('fieldRA', 'f8'),
                                ('fieldDec', 'f8'),
                                ('lst', 'f8')])
    with gzip.open('indata/output_opsim3_61.dat.gz', 'r') as f:
        with astropy.utils.console.Spinner('Loading OpSim data', 'green', step=10000) as sp:
            for i, l in enumerate(f):
                sp.next()
                if i == 0: continue # skip header
                s = l.split()
                fieldID = int(s[3])
                filt = s[4]
                airmass = float(s[25])
                z_a2 = numpy.arccos(1./airmass)
                RA = float(s[28])
                DEC = float(s[29])
                LST = float(s[30])
                out[i-1] = (fieldID, filt, airmass, numpy.nan, z_a2,
                            numpy.nan, numpy.nan, RA, DEC, LST)
    out = out[0:i] # trim
    print 'computing zenith and parallactic angles'
    HA = out['lst'] - out['fieldRA']
    out['HA'] = HA
    out['z_a'], out['q'] = parallactic_zenith_angles(HA, out['fieldDec'])
    return out

def loadcat():
    return numpy.load('indata/opsim.npy')

def lensing_visits(cat):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    X_cond = cat['airmass'] < 2.0
    return numpy.logical_and(numpy.logical_or(r_cond, i_cond), X_cond)

def airmass_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = numpy.arange(-90, 20.001, 2.5) * numpy.pi/180
    X_bins = numpy.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = numpy.histogram2d(cat[w]['airmass'], cat[w]['fieldDec'],
                                          bins=(X_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('airmass')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min(), yedges.max()], aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/airmass_dec_density.pdf')

def airmass_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = numpy.arange(-1.0, 0.3, 0.04)
    X_bins = numpy.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = numpy.histogram2d(cat[w]['airmass'],
                                          numpy.sin(cat[w]['fieldDec']),
                                          bins=(X_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('sin(declination)')
    ax.set_ylabel('airmass')
    im = ax.imshow(H, aspect='auto',
                   extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/airmass_sindec_density.pdf')

def zenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = numpy.arange(-90, 20.001, 2.5) * numpy.pi/180
    z_bins = numpy.arange(0.0, 60.0, 1.0) * numpy.pi/180
    H, yedges, xedges = numpy.histogram2d(cat[w]['z_a'], cat[w]['fieldDec'],
                                          bins=(z_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('zenith angle')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min()*180/numpy.pi, yedges.max()*180/numpy.pi],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/zenith_dec_density.pdf')

def zenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = numpy.arange(-1.0, 0.3, 0.04)
    z_bins = numpy.arange(0.0, 60.0, 1.0) * numpy.pi/180
    H, yedges, xedges = numpy.histogram2d(cat[w]['z_a'], numpy.sin(cat[w]['fieldDec']),
                                          bins=(z_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('sin(declination)')
    ax.set_ylabel('zenith angle')
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min()*180/numpy.pi, yedges.max()*180/numpy.pi],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/zenith_sindec_density.pdf')

def tanzenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = numpy.arange(-90, 20.001, 2.5) * numpy.pi/180
    tanz_bins = numpy.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = numpy.histogram2d(numpy.tan(cat[w]['z_a']), cat[w]['fieldDec'],
                                          bins=(tanz_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('tan(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/tanzenith_dec_density.pdf')

def tanzenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = numpy.arange(-1.0, 0.3, 0.04)
    tanz_bins = numpy.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = numpy.histogram2d(numpy.tan(cat[w]['z_a']), numpy.sin(cat[w]['fieldDec']),
                                          bins=(tanz_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('sin(declination)')
    ax.set_ylabel('tan(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/tanzenith_sindec_density.pdf')

def tan2zenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = numpy.arange(-90, 20.001, 2.5) * numpy.pi/180
    tan2z_bins = numpy.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = numpy.histogram2d(numpy.tan(cat[w]['z_a'])**2, cat[w]['fieldDec'],
                                          bins=(tan2z_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('tan$^2$(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/tan2zenith_dec_density.pdf')

def tan2zenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = numpy.arange(-1.0, 0.3, 0.04)
    tan2z_bins = numpy.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = numpy.histogram2d(numpy.tan(cat[w]['z_a'])**2, numpy.sin(cat[w]['fieldDec']),
                                          bins=(tan2z_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('sin(declination)')
    ax.set_ylabel('tan$^2$(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/tan2zenith_sindec_density.pdf')

def coszenithm35_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = numpy.arange(-90, 20.001, 2.5) * numpy.pi/180
    coszm35_bins = numpy.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = numpy.histogram2d(numpy.cos(cat[w]['z_a'])**(-3/5),
                                          cat[w]['fieldDec'],
                                          bins=(coszm35_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('cos$^{-3/5}$(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/coszenithm35_dec_density.pdf')


def coszenithm35_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = numpy.arange(-1.0, 0.3, 0.04)
    coszm35_bins = numpy.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = numpy.histogram2d(numpy.cos(cat[w]['z_a'])**(-3/5),
                                          numpy.sin(cat[w]['fieldDec']),
                                          bins=(coszm35_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('sin(declination)')
    ax.set_ylabel('cos$^{-3/5}$(zenith angle)')
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect='auto')
    cb = f.colorbar(im)
    cb.set_label('visits')
    if hardcopy:
        f.savefig('output/coszenithm35_sindec_density.pdf')

def airmass_density(cat, hardcopy=False):
    w = lensing_visits(cat)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('airmass')
    ax.set_ylabel('visits')
    ax.set_xlim(1.0, 1.8)
    ax.set_ylim(0, 40000)
    ax.hist(cat[w]['airmass'], bins=100)
    if hardcopy:
        f.savefig('output/airmass_density.pdf')

def angle_dist(cat, fieldID, framenum=None, hardcopy=False):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    f_cond = cat['fieldID'] == fieldID
    w = numpy.where(numpy.logical_and(f_cond, numpy.logical_or(r_cond, i_cond)))[0]
    f = plt.figure()
    ax = f.add_axes([0.1, 0.11, 0.64, 0.79])
#    ax = f.add_subplot(111)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    title = 'Field # = {:04d} $\\alpha$ = {:9.5f}, $\delta$ = {:9.5f}'
    title = title.format(fieldID,
                         cat[w[0]]['fieldRA'] * 180/numpy.pi,
                         cat[w[0]]['fieldDec'] * 180/numpy.pi)
    ax.set_title(title, family='monospace')
    ax.plot([-90., 90], [0.0, 0.0], color='k')
    ax.plot([0.0, 0.0], [-90., 90], color='k')

    # plot curve of possibilities
    HAs = numpy.linspace(0, 2*numpy.pi, 200)
    z, q = parallactic_zenith_angles(HAs, cat[w[0]]['fieldDec'])
    x = z * numpy.cos(q) * 180 / numpy.pi
    y = z * numpy.sin(q) * 180 / numpy.pi
    ax.plot(x, y, color='k', lw=2)

    # plot actual observations color coded by hour angle
    x = cat[w]['z_a'] * numpy.cos(cat[w]['q']) * 180 / numpy.pi
    y = cat[w]['z_a'] * numpy.sin(cat[w]['q']) * 180 / numpy.pi
    h = cat[w]['HA']
    h[h > numpy.pi] = h[h > numpy.pi] - 2.0 * numpy.pi
    h[h < -numpy.pi] = h[h < -numpy.pi] + 2.0 * numpy.pi
    for x1, y1, h1 in zip(x, y, h):
        ax.scatter(x1, y1, color=mpl.cm.rainbow(h1/2/numpy.pi + 0.5), zorder=3)

    f.subplots_adjust(hspace=0.15, wspace=0.07, bottom=0.11, right=0.8)
    cbar_ax = f.add_axes([0.82, 0.11, 0.04, 0.79])
    cbar_ax.set_title('HA')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=-12, vmax=12)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')

    # gray out horizon
    thetas = numpy.linspace(0, numpy.pi, 100)
    x = numpy.cos(thetas)
    y = numpy.sin(thetas)
    ax.fill_between(x * 90, y * 90, 90, color='k', alpha=0.5)
    ax.fill_between(x * 90, -90, -y * 90, color='k', alpha=0.5)

    # plot some zenith angle circles
    thetas = numpy.linspace(0, 2 * numpy.pi, 100)
    x = numpy.cos(thetas)
    y = numpy.sin(thetas)
    ax.plot(x * 60, y * 60, color='k')
    ax.plot(x * 30, y * 30, color='k')

    if hardcopy == True:
        f.savefig('frames/frame{:04d}.png'.format(framenum))
    return framenum

def make_movie_frames(cat, start=0):
    s=list(set(cat['fieldID']))
    pool = mp.Pool(processes=4, maxtasksperchild=4)
    def updatebar(fieldID):
        print fieldID
    for i, s1 in enumerate(s[start:]):
        pool.apply_async(angle_dist, args=(cat, s1, i+start, True), callback=updatebar)
    pool.close()
    pool.join()

def reframe():
    fns = glob.glob('frames/frame????.png')
    numpy.sort(fns)
    for i, fn in enumerate(fns):
        if fn != 'frames/frame{:04d}.png'.format(i):
            os.rename(fn, 'frames/frame{:04d}.png'.format(i))

if __name__ == '__main__':
    cat = loadcat()
    if len(sys.argv) < 2:
        start=0
    else:
        start=int(sys.argv[1])
    make_movie_frames(cat, start)
