import _mypath
import numpy as np

def test_AtmDispPSF():
    from chroma.PSF_model import AtmDispPSF
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
    R = Rrel * plate_scale + atmdisp.get_refraction(685, zenith=zenith)*206265
    m = atmdisp.disp_moments(wave, photons, zenith=zenith)
    print('{:15s} {:15s} {:15s}'.format(' ', 'first moment', 'second moment'))
    print('{:15s} {:15.7f} {:15.7f}'.format('simulated', R, V))
    print('{:15s} {:15.7f} {:15.7f}'.format('analytic', m[0]*206265, m[1]*206265**2))
    print('{:15s} {:15.7f} {:15.7f}'.format('difference', R - m[0]*206265, V - m[1]*206265**2))

def test_ConvolvePSF():
    from chroma import atmdisp
    from chroma.PSF_model import MoffatPSF, AtmDispPSF, ConvolvePSF

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
    m = m[0] - atmdisp.get_refraction(685.0, zenith=zenith), m[1]
    print('change in moments when dispersion is added')
    print('{:15s} {:15s} {:15s}'.format(' ', 'first moment', 'second moment'))
    print('{:15s} {:15.7f} {:15.7f}'.format('simulated',
                                            (ybar-ybar0) * plate_scale,
                                            (Vy-Vy0) * plate_scale**2))
    print('{:15s} {:15.7f} {:15.7f}'.format('analytic', m[0]*206265, m[1]*206265**2))
    print('{:15s} {:15.7f} {:15.7f}'.format('difference',
                                            (ybar-ybar0) * plate_scale - m[0]*206265,
                                            (Vy-Vy0) * plate_scale**2 - m[1]*206265**2))

if __name__ == '__main__':
    test_AtmDispPSF()
    test_ConvolvePSF()
