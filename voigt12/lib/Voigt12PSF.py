from sersic import Sersic
from scipy.integrate import trapz
import numpy as np
from astropy.utils.console import ProgressBar
import hashlib

class Voigt12PSF(object):
    '''Class to handle the Euclid-like chromatic PSF defined in the Voigt+12 color gradient paper.'''
    def __init__(self, wave, photons, ellipticity=0.0, phi=0.0, y0=0.0, x0=0.0):
        '''Initialize a Voigt12PSF instance.

        Arguments
        ---------
        wave -- wavelengths in nm of spectrum
        photons -- d(photons)/d(lambda) spectrum.  Normalization doesn't matter.
        ellipticity -- defined as (a-b)/(a+b) in Voigt+12. (default 0.0)
        phi -- position angle of ellipticity CCW from x-axis. (default 0.0)
        x0, y0 -- center of PSF

        Returns
        -------
        A callable class instance.
        '''
        self.wave = wave
        self.photons = photons / trapz(photons, wave)
        self.ellipticity = ellipticity
        self.phi = phi
        self.y0 = y0
        self.x0 = x0
        self._monochromatic_PSFs = None
        self.key = self.hash()

    @staticmethod
    def _rp(wave):
        '''Effective radius of PSF at given wavelength -- Equation 2 from Voigt+12'''
        rp0 = 0.7
        wave0 = 520.0
        return rp0 * (wave / wave0)**0.6

    def _monochromatic_PSF(self, y0, x0, wave, ellipticity, phi, norm):
        '''Returns the Voigt+12 Euclid Gaussian PSF for particular wavelength.'''
        n = 0.5 # Gaussian
        return Sersic(y0, x0, n, r_e=self._rp(wave), flux=norm, gmag=ellipticity, phi=phi)

    def _load_monochromatic_PSFs(self):
        # create all the monochromatic Gaussians (as Sersics) at initialization and store for later
        print "Loading PSF"
        self._monochromatic_PSFs = []
        with ProgressBar(len(self.wave)) as bar:
            for wav, phot in zip(self.wave, self.photons):
                self._monochromatic_PSFs.append(
                    self._monochromatic_PSF(self.y0, self.x0, wav, self.ellipticity, self.phi, phot))
                bar.update()

    def hash(self):
        '''Make object parameters hashable so a sophisticated calling class won't need to regenerate
        monochromatic PSFs or PSF images unnecessarily if it sees it already has a psf image for
        the same PSF params in its database (see VoigtImageFactory).  Hashing by input parameters is
        better than hashing by the instance ID since more than one instance can be created with the
        same parameters but will have different IDs.

        Somewhat experimental, since I'm no expert on md5 hashes and there are some warnings on
        the intartubze about hashing numpy.ndarray objects (which is why I tupled them)...  but
        seems to work right now...
        '''
        m = hashlib.md5()
        m.update(str((self.y0, self.x0, self.ellipticity, self.phi)))
        m.update(str(tuple(self.wave)))
        m.update(str(tuple(self.photons)))
        return m.hexdigest()

    def psfcube(self, y, x):
        '''Evaluate monochromatic PSFs to make cube of y, x, lambda'''
        if self._monochromatic_PSFs is None:
            self._load_monochromatic_PSFs()
        if isinstance(y, int) or isinstance(y, float):
            y1 = np.array([y])
            x1 = np.array([x])
        if isinstance(y, list) or isinstance(y, tuple):
            y1 = np.array(y)
            x1 = np.array(x)
        if isinstance(y, np.ndarray):
            y1 = y
            x1 = x
        shape = list(y1.shape)
        shape.append(len(self._monochromatic_PSFs))
        psfcube = np.empty(shape, dtype=np.float64)
        print "Evaluating PSF"
        with ProgressBar(len(self._monochromatic_PSFs)) as bar:
            for i, mpsf in enumerate(self._monochromatic_PSFs):
                psfcube[..., i] = mpsf(y1,x1)
                bar.update()
        return psfcube

    def __call__(self, y, x):
        '''Integrate the psfcube over the lambda direction to get the PSF.'''
        psfcube = self.psfcube(y, x)
        print "Integrating over wavelengths"
        return trapz(psfcube, self.wave)

# should develop some real unit tests here.  like matching the input PSF to the sum of the psfcube
# along the spatial directions...  maybe test that the ellipticity makes sense using FWHM
# calculations?

if __name__ == '__main__':
    # this just tests that it's possible for the code to run, nothing about the accuracy of the code
    sed_file = '../data/SEDs/CWW_E_ext.ascii'
    sed_data = np.genfromtxt(sed_file)
    sed_wave, sed_flux = sed_data[:,0], sed_data[:,1]
    filter_file = "../data/filters/voigt12_350.dat"
    filter_data = np.genfromtxt(filter_file)
    wave, filter_tp = filter_data[:,0], filter_data[:,1]
    sed_flux_int = np.interp(wave, sed_wave, sed_flux)
    photons = sed_flux_int * filter_tp * wave
    photons /= photons.max()
    w = np.where(photons > 1.e-5)[0]
    wave = wave[w.min():w.max()]
    photons = photons[w.min():w.max()]
    v = Voigt12PSF(wave, photons)
