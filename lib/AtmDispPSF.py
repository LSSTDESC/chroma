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
        R660 = atmdisp.atm_refrac(660, **self.kwargs)
        pixels = (R - R660) * 206265 / self.plate_scale
        sort = np.argsort(pixels)
        pixels = pixels[sort]
        angle_dens = angle_dens[sort]
        angle_dens /= simps(angle_dens, pixels)
        PSF = np.interp(y, pixels, angle_dens, left=0.0, right=0.0)
        assert self.xloc in x
        PSF *= (x == self.xloc)
        return PSF
