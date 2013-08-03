import gzip

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
                z_a = numpy.arccos(1./airmass)
                RA = float(s[28])
                DEC = float(s[29])
                LST = float(s[30])
                out[i-1] = (fieldID, filt, airmass, z_a, numpy.nan,
                            numpy.nan, numpy.nan, RA, DEC, LST)
    out = out[0:i] # trim
    print 'computing zenith and parallactic angles'
    HA = out['lst'] - out['fieldRA']
    out['HA'] = HA
    out['z_a'], out['q'] = parallactic_zenith_angles(HA, out['fieldDec'])
    return out

def airmass_vs_decln(cat):
    decs = numpy.arange(-90, 90.0001, 5.0)
    airmass = numpy.empty_like(decs)
    dairmass = numpy.empty_like(decs)
    for i, dec in enumerate(decs):
        # conditions
        c1 = cat['filter'] == 'r'
        c2 = cat['fieldDec'] > (dec - 2.5) * numpy.pi/180
        c3 = cat['fieldDec'] < (dec + 2.5) * numpy.pi/180
        w = numpy.where(numpy.logical_and(numpy.logical_and(c1, c2), c3))[0]
        airmass[i] = numpy.mean(cat[w]['airmass'])
        dairmass[i] = numpy.std(cat[w]['airmass'])
    return decs, airmass, dairmass

def airmass_decl_density(cat):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    X_cond = cat['airmass'] < 2.0
    w = numpy.where(numpy.logical_and(numpy.logical_or(r_cond, i_cond), X_cond))[0]
    dec_bins = numpy.arange(-90, 20.001, 2.0) * numpy.pi/180
    X_bins = numpy.arange(1.0, 2.0, 0.02)
    H, yedges, xedges = numpy.histogram2d(cat[w]['airmass'], cat[w]['fieldDec'],
                                          bins=(X_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel('declination')
    ax.set_ylabel('airmass')
    im = ax.imshow(H, extent=[xedges.min()*180/numpy.pi, xedges.max()*180/numpy.pi,
                              yedges.min(), yedges.max()], aspect='auto')
    f.colorbar(im)

def zenith_decl_density(cat):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    za_cond = cat['z_a'] < 60.0 * numpy.pi/180
    w = numpy.where(numpy.logical_and(numpy.logical_or(r_cond, i_cond), za_cond))[0]
    dec_bins = numpy.arange(-90, 20.001, 2.0) * numpy.pi/180
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
    f.colorbar(im)

def angle_dist(cat, fieldID, hardcopy=False):
    r_cond = cat['filter'] == 'r'
    i_cond = cat['filter'] == 'i'
    f_cond = cat['fieldID'] == fieldID
    w = numpy.where(numpy.logical_and(f_cond, numpy.logical_or(r_cond, i_cond)))[0]
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    title = '$\\alpha$ = {:10.5f}, $\delta$ = {:10.5f}'.format(cat[w[0]]['fieldRA'] * 180/numpy.pi,
                                                               cat[w[0]]['fieldDec'] * 180/numpy.pi)
    ax.set_title(title)
    ax.plot([-90., 90], [0.0, 0.0], color='k')
    ax.plot([0.0, 0.0], [-90., 90], color='k')

    # plot curve of possibilities
    HAs = numpy.linspace(0, 2*numpy.pi, 200)
    z, q = parallactic_zenith_angles(HAs, cat[w[0]]['fieldDec'])
    x = z * numpy.cos(q) * 180 / numpy.pi
    y = z * numpy.sin(q) * 180 / numpy.pi
    ax.plot(x, y, color='k')

    # plot actual observations color coded by hour angle
    x = cat[w]['z_a'] * numpy.cos(cat[w]['q']) * 180 / numpy.pi
    y = cat[w]['z_a'] * numpy.sin(cat[w]['q']) * 180 / numpy.pi
    h = cat[w]['HA']
    h[h > numpy.pi] = h[h > numpy.pi] - 2.0 * numpy.pi
    h[h < -numpy.pi] = h[h < -numpy.pi] + 2.0 * numpy.pi
    for x1, y1, h1 in zip(x, y, h):
        ax.scatter(x1, y1, color=mpl.cm.rainbow(h1/2/numpy.pi + 0.5), zorder=3)

    f.subplots_adjust(hspace=0.02, wspace=0.07, bottom=0.11, right=0.8)
    cbar_ax = f.add_axes([0.85, 0.11, 0.04, 0.79])
    cbar_ax.set_title('HA')
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=-12, vmax=12)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')

    thetas = numpy.linspace(0, numpy.pi, 100)
    x = numpy.cos(thetas) * 90
    y = numpy.sin(thetas) * 90
    ax.fill_between(x, y, 90, color='k', alpha=0.5)
    ax.fill_between(x, -90, -y, color='k', alpha=0.5)

    if hardcopy == True:
        f.savefig('frame{:04d}.png'.format(fieldID))
    f.clf()
    return fieldID


def make_movie_frames(cat):
    s=list(set(cat['fieldID']))
    with astropy.utils.console.ProgressBar(len(s)) as bar:
        for s1 in s:
            proc=mp.Process(target=angle_dist, args=(cat, s1, True))
            proc.daemon=True
            proc.start()
            proc.join()
#            angle_dist(cat, s1, hardcopy=True)
            bar.update()

def log_results(fieldID):
    print fieldID

def make_movie_frames2(cat):
    s=list(set(cat['fieldID']))
    pool = mp.Pool(processes=8)
    for s1 in s:
        junk = pool.apply_async(angle_dist, args=(cat, s1, True), callback = log_results)
    pool.close()
    pool.join()


if __name__ == '__main__':
    cat = numpy.load('opsim.npy')
    make_movie_frames2(cat)
