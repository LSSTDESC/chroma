import hashlib
import numpy as np
import atmdisp
from scipy.integrate import simps

class AtmDispPSF(object):
    def __init__(self, wave, photons, plate_scale=0.2, xloc=0.0, **kwargs):
        self.wave = wave
        self.photons = photons
        self.plate_scale = plate_scale
        self.kwargs = kwargs
        self.xloc = xloc
        self.key = self.hash()

    def hash(self):
        m = hashlib.md5()
        m.update(str(tuple(self.wave)))
        m.update(str(tuple(self.photons)))
        m.update(str(self.plate_scale))
        m.update(str(self.xloc))
        keys = self.kwargs.keys()
        keys.sort()
        for key in keys:
            m.update(str((key, self.kwargs[key])))
        return m.hexdigest()

    def __call__(self, y, x):
        if isinstance(y, int) or isinstance(y, float):
            y1 = np.array([y])
            x1 = np.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = np.array(y)
            x1 = np.array(x)
        if isinstance(y, np.ndarray):
            y1 = y
            x1 = x
        R, angle_dens = atmdisp.wave_dens_to_angle_dens(self.wave, self.photons, **self.kwargs)
        R685 = atmdisp.atm_refrac(685.0, **self.kwargs)
        pixels = (R - R685) * 206265 / self.plate_scale
        sort = np.argsort(pixels)
        pixels = pixels[sort]
        angle_dens = angle_dens[sort]
        angle_dens /= simps(angle_dens, pixels)
        PSF = np.interp(y, pixels, angle_dens, left=0.0, right=0.0)
        minx = abs(self.xloc - x).min()
        assert minx < 1.e-10
        PSF *= (abs(self.xloc - x) < 1.e-10)
        return PSF

if __name__ == '__main__':
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
