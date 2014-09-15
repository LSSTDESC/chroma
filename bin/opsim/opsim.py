import sys
import os
import gzip
import glob

import _mypath
import chroma

import numpy as np

import astropy.utils.console
import matplotlib.pyplot as plt
import matplotlib as mpl

import multiprocessing as mp

def parallactic_zenith_angles(HA, dec):
    # latitude = -(30.0 + 14.0/60 + 26.7/3600) * np.pi / 180.0
    # coslat = np.cos(-latitude)
    # sinlat = np.sin(-latitude)
    # hard coding the above in for speed
    latitude = -0.5278006557724753
    coslat = 0.86391682470652653
    sinlat = -0.50363451032369966

    if isinstance(HA, int) or isinstance(HA, float):
        HA = np.array([HA])
        dec = np.array([dec])
    elif isinstance(HA, list) or isinstance(HA, tuple):
        HA = np.array(HA)
        dec = np.array(dec)
    elif not isinstance(HA, np.ndarray):
        raise TypeError

    cosdec = np.cos(dec)
    sindec = np.sin(dec)
    cosHA = np.cos(HA)
    sinHA = np.sin(HA)

    cosza = cosHA * cosdec * coslat + sindec * sinlat
    za = np.arccos(cosza) # za in [0, pi].  check
    sinza = np.sin(za)
    sinq = -sinHA * coslat / sinza
    cosq = (-sinlat + sindec * cosza) / (cosdec * sinza)
    q = np.arctan2(sinq, cosq)

    return za, q

def opsim_fields():
    f = gzip.open("output_opsim3_61.dat.gz","r")
    titles = f.readline().split()
    vals = f.readline().split()
    ii = range(len(titles))
    for i, title, val in zip(ii, titles, vals):
        print "{:02d} {:25} {}".format(i, title, val)

def opsim_parse():
    n = 10000000 # only need about 3e6 of these in the end
    out = np.empty(n, dtype=[("fieldID", "i4"),
                                ("filter", "a1"),
                                ("airmass", "f8"),
                                ("z_a", "f8"),
                                ("z_a2", "f8"),
                                ("HA", "f8"),
                                ("q", "f8"),
                                ("fieldRA", "f8"),
                                ("fieldDec", "f8"),
                                ("lst", "f8"),
                                ("propID", "i4")])
    with gzip.open("indata/output_opsim3_61.dat.gz", "r") as f:
        with astropy.utils.console.Spinner("Loading OpSim data", "green", step=10000) as sp:
            for i, l in enumerate(f):
                sp.next()
                if i == 0: continue # skip header
                s = l.split()
                fieldID = int(s[3])
                filt = s[4]
                airmass = float(s[25])
                z_a2 = np.arccos(1./airmass)
                RA = float(s[28])
                DEC = float(s[29])
                LST = float(s[30])
                propID = int(s[2])
                out[i-1] = (fieldID, filt, airmass, np.nan, z_a2,
                            np.nan, np.nan, RA, DEC, LST, propID)
    out = out[0:i] # trim
    print "computing zenith and parallactic angles"
    HA = out["lst"] - out["fieldRA"]
    out["HA"] = HA
    out["z_a"], out["q"] = parallactic_zenith_angles(HA, out["fieldDec"])
    return out

def loadcat():
    return np.load("indata/opsim.npy")

def savecat(cat):
    np.save("indata/opsim.npy", cat)

def lensing_visits(cat):
    r_cond = cat["filter"] == "r"
    i_cond = cat["filter"] == "i"
    # X_cond = cat["airmass"] < 2.0
    X_cond = (cat["propID"] == 215) | (cat["propID"] == 216) | (cat["propID"] == 218)
    return (r_cond | i_cond) & X_cond

def airmass_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = np.arange(-90, 20.001, 2.5) * np.pi/180
    X_bins = np.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = np.histogram2d(cat[w]["airmass"], cat[w]["fieldDec"],
                                          bins=(X_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("declination")
    ax.set_ylabel("airmass")
    im = ax.imshow(H, extent=[xedges.min()*180/np.pi, xedges.max()*180/np.pi,
                              yedges.min(), yedges.max()], aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/airmass_dec_density.pdf")

def airmass_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = np.arange(-1.0, 0.3, 0.04)
    X_bins = np.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = np.histogram2d(cat[w]["airmass"],
                                          np.sin(cat[w]["fieldDec"]),
                                          bins=(X_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("airmass")
    im = ax.imshow(H, aspect="auto",
                   extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()])
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/airmass_sindec_density.pdf")

def zenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = np.arange(-90, 20.001, 2.5) * np.pi/180
    z_bins = np.arange(0.0, 60.0, 1.0) * np.pi/180
    H, yedges, xedges = np.histogram2d(cat[w]["z_a"], cat[w]["fieldDec"],
                                          bins=(z_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("declination")
    ax.set_ylabel("zenith angle")
    im = ax.imshow(H, extent=[xedges.min()*180/np.pi, xedges.max()*180/np.pi,
                              yedges.min()*180/np.pi, yedges.max()*180/np.pi],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/zenith_dec_density.pdf")

def zenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = np.arange(-1.0, 0.3, 0.04)
    z_bins = np.arange(0.0, 60.0, 1.0) * np.pi/180
    H, yedges, xedges = np.histogram2d(cat[w]["z_a"], np.sin(cat[w]["fieldDec"]),
                                          bins=(z_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("zenith angle")
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min()*180/np.pi, yedges.max()*180/np.pi],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/zenith_sindec_density.pdf")

def tanzenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = np.arange(-90, 20.001, 2.5) * np.pi/180
    tanz_bins = np.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = np.histogram2d(np.tan(cat[w]["z_a"]), cat[w]["fieldDec"],
                                          bins=(tanz_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("declination")
    ax.set_ylabel("tan(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min()*180/np.pi, xedges.max()*180/np.pi,
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/tanzenith_dec_density.pdf")

def tanzenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = np.arange(-1.0, 0.3, 0.04)
    tanz_bins = np.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = np.histogram2d(np.tan(cat[w]["z_a"]), np.sin(cat[w]["fieldDec"]),
                                          bins=(tanz_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("tan(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/tanzenith_sindec_density.pdf")

def tan2zenith_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = np.arange(-90, 20.001, 2.5) * np.pi/180
    tan2z_bins = np.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = np.histogram2d(np.tan(cat[w]["z_a"])**2, cat[w]["fieldDec"],
                                          bins=(tan2z_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("declination")
    ax.set_ylabel("tan$^2$(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min()*180/np.pi, xedges.max()*180/np.pi,
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/tan2zenith_dec_density.pdf")

def tan2zenith_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = np.arange(-1.0, 0.3, 0.04)
    tan2z_bins = np.arange(0.0, 1.5, 0.025)
    H, yedges, xedges = np.histogram2d(np.tan(cat[w]["z_a"])**2, np.sin(cat[w]["fieldDec"]),
                                          bins=(tan2z_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("tan$^2$(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/tan2zenith_sindec_density.pdf")

def coszenithm35_dec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    dec_bins = np.arange(-90, 20.001, 2.5) * np.pi/180
    coszm35_bins = np.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = np.histogram2d(np.cos(cat[w]["z_a"])**(-3/5),
                                          cat[w]["fieldDec"],
                                          bins=(coszm35_bins, dec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("declination")
    ax.set_ylabel("cos$^{-3/5}$(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min()*180/np.pi, xedges.max()*180/np.pi,
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/coszenithm35_dec_density.pdf")


def coszenithm35_sindec_density(cat, hardcopy=False):
    w = lensing_visits(cat)

    sindec_bins = np.arange(-1.0, 0.3, 0.04)
    coszm35_bins = np.arange(1.0, 1.6, 0.01)
    H, yedges, xedges = np.histogram2d(np.cos(cat[w]["z_a"])**(-3/5),
                                          np.sin(cat[w]["fieldDec"]),
                                          bins=(coszm35_bins, sindec_bins))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("sin(declination)")
    ax.set_ylabel("cos$^{-3/5}$(zenith angle)")
    im = ax.imshow(H, extent=[xedges.min(), xedges.max(),
                              yedges.min(), yedges.max()],
                              aspect="auto")
    cb = f.colorbar(im)
    cb.set_label("visits")
    if hardcopy:
        f.savefig("output/coszenithm35_sindec_density.pdf")

def airmass_density(cat, hardcopy=False):
    w = lensing_visits(cat)
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.set_xlabel("airmass")
    ax.set_ylabel("visits")
    ax.set_xlim(1.0, 1.6)
    ax.set_ylim(0, 60000)
    ax.hist(cat[w]["airmass"], bins=100, histtype="stepfilled")
    if hardcopy:
        f.savefig("output/airmass_density.pdf")

def angle_dist(cat, fieldID, framenum=None, hardcopy=False):
    r_cond = cat["filter"] == "r"
    i_cond = cat["filter"] == "i"
    f_cond = cat["fieldID"] == fieldID
    w = np.where(np.logical_and(f_cond, np.logical_or(r_cond, i_cond)))[0]
    f = plt.figure()
    ax = f.add_axes([0.1, 0.11, 0.64, 0.79])
#    ax = f.add_subplot(111)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    title = "Field # = {:04d} $\\alpha$ = {:9.5f}, $\delta$ = {:9.5f}"
    title = title.format(fieldID,
                         cat[w[0]]["fieldRA"] * 180/np.pi,
                         cat[w[0]]["fieldDec"] * 180/np.pi)
    ax.set_title(title, family="monospace")
    ax.plot([-90., 90], [0.0, 0.0], color="k")
    ax.plot([0.0, 0.0], [-90., 90], color="k")

    # plot curve of possibilities
    HAs = np.linspace(0, 2*np.pi, 200)
    z, q = parallactic_zenith_angles(HAs, cat[w[0]]["fieldDec"])
    x = z * np.cos(q) * 180 / np.pi
    y = z * np.sin(q) * 180 / np.pi
    ax.plot(x, y, color="k", lw=2)

    # plot actual observations color coded by hour angle
    x = cat[w]["z_a"] * np.cos(cat[w]["q"]) * 180 / np.pi
    y = cat[w]["z_a"] * np.sin(cat[w]["q"]) * 180 / np.pi
    h = cat[w]["HA"]
    h[h > np.pi] = h[h > np.pi] - 2.0 * np.pi
    h[h < -np.pi] = h[h < -np.pi] + 2.0 * np.pi
    for x1, y1, h1 in zip(x, y, h):
        ax.scatter(x1, y1, color=mpl.cm.rainbow(h1/2/np.pi + 0.5), zorder=3)

    f.subplots_adjust(hspace=0.15, wspace=0.07, bottom=0.11, right=0.8)
    cbar_ax = f.add_axes([0.82, 0.11, 0.04, 0.79])
    cbar_ax.set_title("HA")
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=-12, vmax=12)
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")

    # gray out horizon
    thetas = np.linspace(0, np.pi, 100)
    x = np.cos(thetas)
    y = np.sin(thetas)
    ax.fill_between(x * 90, y * 90, 90, color="k", alpha=0.5)
    ax.fill_between(x * 90, -90, -y * 90, color="k", alpha=0.5)

    # plot some zenith angle circles
    thetas = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(thetas)
    y = np.sin(thetas)
    ax.plot(x * 60, y * 60, color="k")
    ax.plot(x * 30, y * 30, color="k")

    if hardcopy == True:
        f.savefig("frames/frame{:04d}.png".format(framenum))
    return framenum

def epoch_variance(cat):
    good = lensing_visits(cat)
    dec_edges = np.arange(-66.0, 6.0, 1.0) * np.pi/180
    decs = (dec_edges[1:] + dec_edges[:-1])*0.5 * 180/np.pi
    dxs = []
    dys = []
    dxys = []
    for i in range(len(dec_edges)-1):
        w = (cat["fieldDec"] > dec_edges[i]) & (cat["fieldDec"] < dec_edges[i+1]) & good
        mean_zenith = np.mean(cat[w]["z_a"])
        x0 = np.tan(mean_zenith)*np.sin(cat[w]["q"])
        y0 = np.tan(mean_zenith)*np.cos(cat[w]["q"])
        centroid = (np.mean(x0), np.mean(y0))
        dxs.append(np.mean((x0-centroid[0])**2))
        dys.append(np.mean((y0-centroid[1])**2))
        dxys.append(np.mean((x0-centroid[0])*(y0-centroid[1])))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(decs, dxs, color="red", label=r"$\langle(\Delta x)^2\rangle_\mathrm{epochs}$")
    ax.plot(decs, dys, color="blue", label=r"$\langle(\Delta y)^2\rangle_\mathrm{epochs}$")
    ax.plot(decs, dxys, color="green", label=r"$\langle(\Delta x)(\Delta y)\rangle_\mathrm{epochs}$")
    ax.legend()
    ax.set_xlabel('Declination (deg)')
    ax.set_ylabel('Misregistration second moments (arcsec$^2$)')
    plt.savefig('output/epoch_variance.png', dpi=220)

def epoch_variance_bias(cat):
    good = lensing_visits(cat)
    dec_edges = np.arange(-66.0, 6.0, 1.0) * np.pi/180
    decs = (dec_edges[1:] + dec_edges[:-1])*0.5 * 180/np.pi
    dxs = []
    dys = []
    dxys = []
    ms = []
    c1s = []
    c2s = []
    rsquared_gal = 0.4**2
    for i in range(len(dec_edges)-1):
        w = (cat["fieldDec"] > dec_edges[i]) & (cat["fieldDec"] < dec_edges[i+1]) & good
        x0 = np.tan(cat[w]['z_a'])*np.sin(cat[w]["q"])
        y0 = np.tan(cat[w]['z_a'])*np.cos(cat[w]["q"])
        centroid = (np.mean(x0), np.mean(y0))
        dxs.append(np.mean((x0-centroid[0])**2))
        dys.append(np.mean((y0-centroid[1])**2))
        dxys.append(np.mean((x0-centroid[0])*(y0-centroid[1])))
        ms.append((-dxs[-1]-dys[-1])/rsquared_gal)
        c1s.append((dxs[-1]-dys[-1])/(2.0*rsquared_gal))
        c2s.append((dxys[-1])/rsquared_gal)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(decs, ms, color="red", label=r"$m / (\Delta \bar{R}_{45})^2$")
    ax.plot(decs, c1s, color="blue", label=r"$c_+ / (\Delta \bar{R}_{45})^2$")
    ax.plot(decs, c2s, color="green", label=r"$c_\times / (\Delta \bar{R}_{45})^2$")
    ax.legend()
    ax.set_xlabel('Declination (deg)')
    ax.set_ylabel(r"Shear calibration bias / ($\Delta \bar{R}_{45})^2$ (arcsec$^{-2}$)")
    ax.set_ylim(-2.0, 1.0)
    plt.savefig('output/misregistration_bias.png', dpi=220)

def epoch_variance_bias_fields(cat):
    good = lensing_visits(cat)
    dxs = []
    dys = []
    dxys = []
    ms = []
    c1s = []
    c2s = []
    decs = []
    fields = np.unique(cat['fieldID'])
    rsquared_gal = 0.4**2
    with chroma.ProgressBar(len(fields)) as bar:
        for field in fields:
            bar.update()
            w = (cat['fieldID'] == field) & good
            if w.sum() < 100: continue
            x0 = np.tan(cat[w]['z_a'])*np.sin(cat[w]["q"])
            y0 = np.tan(cat[w]['z_a'])*np.cos(cat[w]["q"])
            centroid = (np.mean(x0), np.mean(y0))
            dxs.append(np.mean((x0-centroid[0])**2))
            dys.append(np.mean((y0-centroid[1])**2))
            dxys.append(np.mean((x0-centroid[0])*(y0-centroid[1])))
            ms.append((-dxs[-1]-dys[-1])/rsquared_gal)
            c1s.append((dxs[-1]-dys[-1])/(2.0*rsquared_gal))
            c2s.append((dxys[-1])/rsquared_gal)
            decs.append(cat[w]['fieldDec'][0] * 180/np.pi)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111)
    ax.scatter(decs, ms, s=5, color="red", label=r"$m / (\Delta \bar{R}_{45})^2$")
    ax.scatter(decs, c1s, s=5, color="blue", label=r"$c_+ / (\Delta \bar{R}_{45})^2$")
    ax.scatter(decs, c2s, s=5, color="green", label=r"$c_\times / (\Delta \bar{R}_{45})^2$")
    ax.plot(decs, [0]*len(decs), color='black')
    ax.legend(prop={'size':14})
    ax.set_xlabel('Declination (deg)', fontsize=14)
    ax.set_ylabel(r"Shear calibration bias / ($\Delta \bar{R}_{45})^2$ (arcsec$^{-2}$)", fontsize=14)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xlim(-70, 10)
    fig.tight_layout()
    if not os.path.isdir('output'):
        os.mkdir('output')
    plt.savefig('output/misregistration_bias_fields.png', dpi=220)

def epoch_variance_field(cat, field):
    good = lensing_visits(cat)
    w = (cat['fieldID'] == field) & good
    x = np.tan(cat[w]['z_a']) * np.sin(cat[w]['q'])
    y = np.tan(cat[w]['z_a']) * np.cos(cat[w]['q'])
    centroid = (np.mean(x), np.mean(y))
    dx = np.mean((x-centroid[0])**2)
    dy = np.mean((y-centroid[1])**2)
    dxy = np.mean((x-centroid[0])*(y-centroid[1]))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=5)
    ax.scatter(centroid[0], centroid[1], marker='x', color='red', s=80)
    ax.set_xlabel(r"$\cos\delta\, \Delta \alpha / \Delta \bar{R}_{45}$", fontsize=14)
    ax.set_ylabel(r"$\Delta \delta / \Delta \bar{R}_{45}$", fontsize=14)
    ax.text(0.4, 0.9, r"$\delta$ = {:5.3f}$^\circ$".format(cat[w]['fieldDec'][0] * 180/np.pi),
            transform=ax.transAxes, fontsize=14)
    ax.text(0.4, 0.83, r"$\alpha$ = {:5.3f}$^\circ$".format(cat[w]['fieldRA'][0] * 180/np.pi),
            transform=ax.transAxes, fontsize=14)
    ax.text(0.4, 0.76, r"$\langle(\cos \delta\, \Delta\alpha)^2\rangle/(\Delta \bar{R}_{45})^2$ = "+
                       "{:5.3f}".format(dx),
                       transform=ax.transAxes, fontsize=14)
    ax.text(0.4, 0.69, r"$\langle(\Delta\delta)^2\rangle/(\Delta \bar{R}_{45})^2$ = "+
                       "{:5.3f}".format(dy),
                       transform=ax.transAxes, fontsize=14)
    ax.text(0.4, 0.62, r"$\langle(\cos \delta\, \Delta\alpha) (\Delta \delta)\rangle/(\Delta\bar{R}_{45})^2$ = "+
                       "{:5.3f}".format(dxy),
                       transform=ax.transAxes, fontsize=14)
    fig.tight_layout()
    if not os.path.isdir('output'):
        os.mkdir('output')
    plt.savefig('output/epoch_variance_field{}.png'.format(field), dpi=220)

def make_movie_frames(cat, start=0):
    s=list(set(cat["fieldID"]))
    pool = mp.Pool(processes=4, maxtasksperchild=4)
    def updatebar(fieldID):
        print fieldID
    for i, s1 in enumerate(s[start:]):
        pool.apply_async(angle_dist, args=(cat, s1, i+start, True), callback=updatebar)
    pool.close()
    pool.join()

def reframe():
    fns = glob.glob("frames/frame????.png")
    np.sort(fns)
    for i, fn in enumerate(fns):
        if fn != "frames/frame{:04d}.png".format(i):
            os.rename(fn, "frames/frame{:04d}.png".format(i))

if __name__ == "__main__":
    cat = loadcat()
    if len(sys.argv) < 2:
        start=0
    else:
        start=int(sys.argv[1])
    make_movie_frames(cat, start)
