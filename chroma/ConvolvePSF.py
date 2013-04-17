import numpy as np
import hashlib
from scipy.signal import fftconvolve

class ConvolvePSF(object):
    def __init__(self, PSFs, factor=3):
        self.PSFs = PSFs
        self.factor = factor
        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        for PSF in self.PSFs:
            m.update(PSF.key)
        return m.hexdigest()

    @staticmethod
    def _rebin(a, shape):
        '''Bin down image a to have final size given by shape.

        I think I stole this from stackoverflow somewhere...
        '''
        sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    def __call__(self, y, x):
        # first compute `factor` oversampled coordinates from y, x, which are assumed to be in a
        # grid and uniformly spaced (any way to relax this assumption?)
        if isinstance(y, int) or isinstance(y, float):
            y1 = np.array([y])
            x1 = np.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = np.array(y)
            x1 = np.array(x)
        if isinstance(y, np.ndarray):
            y1 = y
            x1 = x
        nx = x.shape[1]
        ny = y.shape[0]
        dx = (x.max() - x.min())/(nx - 1.0)
        dy = (y.max() - y.min())/(ny - 1.0)
        x0 = x.min() - 0.5 * dx
        y0 = y.min() - 0.5 * dy
        x1 = x.max() + 0.5 * dx
        y1 = y.max() + 0.5 * dy
        dsubx = dx / self.factor
        dsuby = dy / self.factor
        xsub = np.linspace(x0 + dsubx/2.0, x1 - dsubx/2.0, nx * self.factor)
        ysub = np.linspace(y0 + dsubx/2.0, y1 - dsubx/2.0, ny * self.factor)
        xsub, ysub = np.meshgrid(xsub, ysub)

        over = self.PSFs[0](ysub, xsub)
        for PSF in self.PSFs[1:]:
            over = fftconvolve(over, PSF(ysub, xsub), mode='same')
        return self._rebin(over, x.shape)

if __name__ == '__main__':
    import MoffatPSF
    import AtmDispPSF
    import atmdisp

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

    aPSF = AtmDispPSF.AtmDispPSF(wave, photons, zenith=zenith, plate_scale=plate_scale)
    mPSF = MoffatPSF.MoffatPSF(0.0, 0.0, beta=4.6, flux=1.0, FWHM=0.6 / plate_scale,
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
