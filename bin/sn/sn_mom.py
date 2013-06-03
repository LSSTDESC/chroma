import numpy

import _mypath
import chroma.atmdisp

def moments(s_wave, s_flux, f_wave, f_throughput, zenith, **kwargs):
    wave = f_wave[f_wave > 300]
    flambda_i = numpy.interp(wave, s_wave, s_flux)
    throughput_i = numpy.interp(wave, f_wave, f_throughput)
    photons = flambda_i * throughput_i * wave

    m = chroma.atmdisp.disp_moments(wave, photons, zenith=zenith * numpy.pi / 180.0, **kwargs)
    return m
    # gaussian_sigma = 1.0 / 2.35 # 1 arcsec FWHM -> sigma
    # m2 = chroma.atmdisp.weighted_second_moment(wave, photons, 1.0,
    #                                            zenith=zenith * numpy.pi / 180.0,
    #                                            Rbar=m[0], **kwargs)
    # return m[0], m[1], m2

def uniqify(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

def sn_mom(z, filter_name, zenith=30):
    sed_dir = '../../data/SEDs/'
    hsiao_data = numpy.genfromtxt(sed_dir+'hsiao.dat')
    epoch1, wave1, flux1 = hsiao_data[:,0], hsiao_data[:,1], hsiao_data[:,2]
    epoch = numpy.array(uniqify(epoch1))
    nepoch = len(epoch)
    wave = numpy.array(uniqify(wave1)) * 0.1 # Ang -> nm
    nwave = len(wave)
    flux = numpy.reshape(flux1, [nepoch, nwave])

    filter_dir = '../../data/filters/'
    f_data = numpy.genfromtxt(filter_dir+'LSST_{}.dat'.format(filter_name))
    f_wave, f_throughput = f_data[:,0], f_data[:,1]

    wave *= (1.0 + z)

    Rbars = numpy.empty(0)

    for iday in range(len(epoch)):
        m = moments(wave * (1.0 + z), flux[iday,:], f_wave, f_throughput, zenith)
        Rbars = numpy.append(Rbars, m[0])

    return epoch, Rbars
