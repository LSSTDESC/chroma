import _mypath
import numpy as np

def test_Sersic():
    from chroma.SBProfile import Sersic
    '''Creating a test case for the various ways of initializing Sersics'''
    y0 = 0.1
    x0 = 0.3
    n = 1.5
    peak = 1.0

    # a, b, phi
    phi = 0.1
    a = 0.5
    b = 0.2
    s1 = Sersic(y0, x0, n, peak=peak, a=a, b=b, phi=phi)
    print 'It is an error if following 9 or so values do not all match'
    print s1(0.2, 2.1)

    # C11, C12, C22
    C11 = (np.cos(phi)**2.0 / a**2.0 + np.sin(phi)**2.0 / b**2.0)
    C22 = (np.sin(phi)**2.0 / a**2.0 + np.cos(phi)**2.0 / b**2.0)
    C12 = 0.5 * (1.0/a**2.0 - 1.0/b**2.0) * np.sin(2.0 * phi)
    s2 = Sersic(y0, x0, n, peak=peak, C11=C11, C12=C12, C22=C22)
    print s2(0.2, 2.1)

    # r_e, b_over_a, phi
    r_e = np.sqrt(a * b)
    b_over_a = b / a
    s3 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print s3(0.2, 2.1)

    # r_e, emag, phi
    emag = (a**2.0 - b**2.0) / (a**2.0 + b**2.0)
    s4 = Sersic(y0, x0, n, peak=peak, r_e=r_e, emag=emag, phi=phi)
    print s4(0.2, 2.1)

    # r_e, gmag, phi
    gmag = (a - b) / (a + b)
    s5 = Sersic(y0, x0, n, peak=peak, r_e=r_e, gmag=gmag, phi=phi)
    print s5(0.2, 2.1)

    # r_e, e1, e2
    e1 = emag * np.cos(2.0 * phi)
    e2 = emag * np.sin(2.0 * phi)
    s6 = Sersic(y0, x0, n, peak=peak, r_e=r_e, e1=e1, e2=e2)
    print s6(0.2, 2.1)

    # r_e, g1, g2
    g1 = gmag * np.cos(2.0 * phi)
    g2 = gmag * np.sin(2.0 * phi)
    s7 = Sersic(y0, x0, n, peak=peak, r_e=r_e, g1=g1, g2=g2)
    print s7(0.2, 2.1)

    # FWHM instead of r_e
    FWHM = 2 * r_e * (np.log(2.0) / Sersic.compute_kappa(n))**n
    s8 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print s8(0.2, 2.1)

    # flux instead of peak
    flux=Sersic.compute_flux(n, r_e, peak)
    s9 = Sersic(y0, x0, n, flux=flux, a=a, b=b, phi=phi)
    print s9(0.2, 2.1)

    # make sure have access to r_e
    print 'It is an error if following 9 or so values do not all match'
    print s1.r_e
    print s2.r_e
    print s3.r_e
    print s4.r_e
    print s5.r_e
    print s6.r_e
    print s7.r_e
    print s8.r_e
    print s9.r_e


    # make sure have access to a
    print 'It is an error if following 9 or so values do not all match'
    print s1.a
    print s2.a
    print s3.a
    print s4.a
    print s5.a
    print s6.a
    print s7.a
    print s8.a
    print s9.a


    # make sure have access to b
    print 'It is an error if following 9 or so values do not all match'
    print s1.b
    print s2.b
    print s3.b
    print s4.b
    print s5.b
    print s6.b
    print s7.b
    print s8.b
    print s9.b


    # make sure have access to phi
    print 'It is an error if following 9 or so values do not all match'
    print s1.phi
    print s2.phi
    print s3.phi
    print s4.phi
    print s5.phi
    print s6.phi
    print s7.phi
    print s8.phi
    print s9.phi

    # make sure have access to C11
    print 'It is an error if following 9 or so values do not all match'
    print s1.C11
    print s2.C11
    print s3.C11
    print s4.C11
    print s5.C11
    print s6.C11
    print s7.C11
    print s8.C11
    print s9.C11

    # make sure have access to C22
    print 'It is an error if following 9 or so values do not all match'
    print s1.C22
    print s2.C22
    print s3.C22
    print s4.C22
    print s5.C22
    print s6.C22
    print s7.C22
    print s8.C22
    print s9.C22

    # make sure have access to C12
    print 'It is an error if following 9 or so values do not all match'
    print s1.C12
    print s2.C12
    print s3.C12
    print s4.C12
    print s5.C12
    print s6.C12
    print s7.C12
    print s8.C12
    print s9.C12

def test_AtmDispPSF():
    from chroma.SBProfile import AtmDispPSF
    from chroma import atmdisp
    from scipy.integrate import simps

    fdata = np.genfromtxt('../data/filters/LSST_r.dat')
    wave, fthroughput = fdata[:,0], fdata[:,1]
    sdata = np.genfromtxt('../data/SEDs/ukk5v.ascii')
    swave, flux = sdata[:,0], sdata[:,1]
    flux_i = np.interp(wave, swave, flux)
    photons = flux_i * fthroughput * wave

    plate_scale = 0.2
    zenith = 50.*np.pi/180
    over = 21.0

    psf = AtmDispPSF(wave, photons, zenith=zenith, plate_scale=plate_scale)
    x = np.r_[0]
    y = np.arange(15 * over)
    y /= over
    y -= np.median(y)
    x, y = np.meshgrid(x, y)
    psfim = psf(y, x).flatten()
    yflat = y.flatten()
    norm = simps(psfim, yflat)
    Rrel = simps(psfim * yflat, yflat)/norm
    V = simps(psfim * (yflat-Rrel)**2, yflat)/norm * plate_scale**2
    R = Rrel * plate_scale + atmdisp.atm_refrac(685, zenith=zenith)*206265
    m = atmdisp.disp_moments(wave, photons, zenith=zenith)
    print '{:15s} {:15s} {:15s}'.format(' ', 'first moment', 'second moment')
    print '{:15s} {:15.7f} {:15.7f}'.format('simulated', R, V)
    print '{:15s} {:15.7f} {:15.7f}'.format('analytic', m[0]*206265, m[1]*206265**2)
    print '{:15s} {:15.7f} {:15.7f}'.format('difference', R - m[0]*206265, V - m[1]*206265**2)

def test_ConvolvePSF():
    from chroma import atmdisp
    from chroma.SBProfile import MoffatPSF, AtmDispPSF, ConvolvePSF

    fdata = np.genfromtxt('../data/filters/LSST_r.dat')
    sdata = np.genfromtxt('../data/SEDs/CWW_E_ext.ascii')
    plate_scale = 0.2
    zenith = 10.0 * np.pi / 180
    size = 25
    factor1 = 9

    wave, fthroughput = fdata[:,0], fdata[:,1]
    swave, flux = sdata[:,0], sdata[:,1]
    flux_i = np.interp(wave, swave, flux)
    photons = flux_i * fthroughput * wave

    aPSF = AtmDispPSF(wave, photons, zenith=zenith, plate_scale=plate_scale)
    mPSF = MoffatPSF(0.0, 0.0, beta=4.6, flux=1.0, FWHM=0.6 / plate_scale,
                               gmag=0.0, phi=0.0)
    cPSF = ConvolvePSF([aPSF, mPSF])

    oversize = size * factor1
    mn = -size/2.0
    mx = size/2.0
    dsubpix = 1.0/size/factor1
    subpix = np.linspace(mn + dsubpix/2.0, mx - dsubpix/2.0, size * factor1)
    x, y = np.meshgrid(subpix, subpix)

    aPSF_im = aPSF(y, x)

    mPSF_im = mPSF(y, x)
    norm0 = mPSF_im.sum()
    ybar0 = (mPSF_im * y).sum()/norm0
    Vy0 = (mPSF_im * (y-ybar0)**2).sum()/norm0

    cPSF_im = cPSF(y, x)
    norm = cPSF_im.sum()
    ybar = (cPSF_im * y).sum()/norm
    Vy = (cPSF_im * (y-ybar)**2).sum()/norm


    m = atmdisp.disp_moments(wave, photons, zenith=zenith)
    m = m[0] - atmdisp.atm_refrac(685.0, zenith=zenith), m[1]
    print 'change in moments when dispersion is added'
    print '{:15s} {:15s} {:15s}'.format(' ', 'first moment', 'second moment')
    print '{:15s} {:15.7f} {:15.7f}'.format('simulated',
                                            (ybar-ybar0) * plate_scale,
                                            (Vy-Vy0) * plate_scale**2)
    print '{:15s} {:15.7f} {:15.7f}'.format('analytic', m[0]*206265, m[1]*206265**2)
    print '{:15s} {:15.7f} {:15.7f}'.format('difference',
                                            (ybar-ybar0) * plate_scale - m[0]*206265,
                                            (Vy-Vy0) * plate_scale**2 - m[1]*206265**2)

if __name__ == '__main__':
    test_Sersic()
    test_AtmDispPSF()
    test_ConvolvePSF()
