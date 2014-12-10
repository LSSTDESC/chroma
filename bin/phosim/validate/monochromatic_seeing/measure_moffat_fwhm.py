import os
import sys
import cPickle

import numpy as np
import lmfit
import galsim
#import matplotlib.pyplot as plt
from astropy import wcs
from astropy.io import fits

filter_number = {'u':'0','g':'1','r':'2','i':'3','z':'4','Y':'5'}

def encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed):
    mode = 0
    if atm:
        mode += 1
    if telescope:
        mode += 2
    if sensor:
        mode += 4
    mode_digit = str(mode)
    wave_digits = "{:02d}".format(wavelength / 25)
    filter_digit = filter_number[filter_name]
    seed_digits = "{:03d}".format(seed-1000)
    return mode_digit + wave_digits + filter_digit + seed_digits

def resid(params, target):
    shape = target.shape
    center = [0.5*(shape[0]-1), 0.5*(shape[1]-1)]
    #psf = galsim.Moffat(fwhm=params['fwhm'].value, beta=params['beta'].value)
    psf = galsim.Gaussian(fwhm=params['fwhm'].value)
    psf *= params['flux'].value
    psf = psf.shear(e1=params['e1'].value, e2=params['e2'].value)
    psf = psf.shift(params['x0'].value-center[1], params['y0'].value-center[0])
    img = galsim.ImageF(shape[1], shape[0], scale=1.0)
    psf.drawImage(image=img)
    return (img.array - target).ravel()

def moments(data, pixsize=1.0):
    '''Compute first and second (quadrupole) moments of `data`.  Scales result by `pixsize` for
    non-unit width pixels.

    Arguments
    ---------
    data -- array to analyze
    pixsize -- linear size of a pixel
    '''
    ys, xs = np.mgrid[0:data.shape[0], 0:data.shape[1]] * pixsize
    total = data.sum()
    xbar = (data * xs).sum() / total
    ybar = (data * ys).sum() / total
    Ixx = (data * (xs-xbar)**2).sum() / total
    Iyy = (data * (ys-ybar)**2).sum() / total
    Ixy = (data * (xs - xbar) * (ys - ybar)).sum() / total
    return total, xbar, ybar, Ixx, Iyy, Ixy

def moments_to_ellipse(Ixx, Ixy, Iyy):
    r = np.sqrt(np.sqrt(4.0 * (Ixx*Iyy - Ixy**2))) # determinant
    phi = 0.5 * np.arctan2(Iyy - Ixx, 2*Ixy)
    discriminant = 4.0 / r**4 * (Ixx + Iyy)**2 - 4.0
    ab = 0.5 * ((2 / r**2) * (Ixx + Iyy) + np.sqrt(discriminant))
    ba = 1./ab
    return r, ba, phi

def measure_moffat_fwhm(atm, telescope, sensor, wavelength, filter_name, chip, seed, noclobber=False):
    out = {}
    if wavelength == 500:
        return out
    obshistid = encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed)
    outfilename = "output/psf_fit_"+obshistid+str(chip)+".pkl"
    if os.path.isfile(outfilename) and noclobber:
        return
    if chip == 1:
        chipid = "R02_S11"
    elif chip == 4:
        chipid = "R12_S11"
    elif chip == 7:
        chipid = "R22_S11"

    image_file_template = 'output/lsst_e_{}_f{}_{}_E000.fits.gz'
    image_file = image_file_template.format(obshistid, filter_number[filter_name], chipid)

    try:
        hdulist = fits.open(image_file)
    except:
        return out
    w = wcs.WCS(hdulist[0].header)
    data = hdulist[0].data

    RAs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]
    DECs = [-0.035, -0.025, -0.015, -0.005, 0.005, 0.015, 0.025, 0.035]

    values = np.empty(len(RAs) * len(DECs),
                      dtype=[('flux','f4'),
                             ('fwhm','f4'),
                             #('beta','f4'),
                             ('e1','f4'),
                             ('e2','f4'),
                             ('dx','f4'),
                             ('dy','f4')])

    i=0
    for RA in RAs:
        objRA = RA + (chip-7)*0.235
        for DEC in DECs:
            # get x, y
            center = w.wcs_world2pix(np.array([[objRA, DEC]], np.float_),0)[0]
            # reverse to match np's y, x convention
            center = [center[1], center[0]]
            # get the integer pixel index
            center_pix = [int(c) for c in center]
            thumb = data[center_pix[0]-30:center_pix[0]+31,
                         center_pix[1]-30:center_pix[1]+31]
            flux, xbar, ybar, Ixx, Iyy, Ixy = moments(thumb)
            recenter_pix = [center_pix[0] + (ybar-30), center_pix[1] + (xbar-30)]
            rethumb = data[recenter_pix[0]-30:recenter_pix[0]+31,
                           recenter_pix[1]-30:recenter_pix[1]+31]
            flux, xbar, ybar, Ixx, Iyy, Ixy = moments(rethumb)

            if flux == 0.0:
                return out

            r, q, phi = moments_to_ellipse(Ixx, Ixy, Iyy)
            s = galsim.Shear(q=q, beta=phi*galsim.radians)

            params = lmfit.Parameters()
            params.add('flux', value=flux, min=0.0)
            params.add('fwhm', value=r * 0.8, min=0.0, max=5.0)
            #params.add('beta', value=2.7, min=2.0, max=5.0)
            params.add('e1', value=s.e1, max=0.5)
            params.add('e2', value=s.e2, max=0.5)
            params.add('x0', value=xbar)
            params.add('y0', value=ybar)
            result = lmfit.minimize(resid, params, args=(rethumb,))
            print result.redchi

            x0 = (recenter_pix[1]-30) + result.params['x0'].value
            dx = x0 - center[1]
            y0 = (recenter_pix[0]-30) + result.params['y0'].value
            dy = y0 - center[0]

            values[i] = (result.params['flux'].value,
                         result.params['fwhm'].value,
                         #result.params['beta'].value,
                         result.params['e1'].value,
                         result.params['e2'].value,
                         dx,
                         dy)
            i += 1
            print "star: {}  flux: {}  fwhm: {}".format(i, result.params['flux'].value,
                                                        result.params['fwhm'].value)

    print atm, telescope, sensor, wavelength, filter_name, chip
    outstring = 'fwhm = {:5.3f} +/- {:5.3f}    flux = {:010f}'
    print outstring.format(np.mean(values['fwhm']),
                           np.std(values['fwhm']),
                           np.mean(values['flux']))

    #for item in ['flux', 'fwhm', 'beta', 'e1', 'e2', 'dx', 'dy']:
    for item in ['flux', 'fwhm', 'e1', 'e2', 'dx', 'dy']:
        out[item] = np.mean(values[item])
        out[item+'_err'] = np.std(values[item])

    cPickle.dump(out, open(outfilename, 'wb'))

def submit_job(atm, telescope, sensor, wavelength, filter_name, chip, seed, noclobber=False):
    import subprocess
    if wavelength == 500:
        return
    obshistid = encode_obshistid(atm, telescope, sensor, wavelength, filter_name, seed)
    outfilename = "output/psf_fit_"+obshistid+str(chip)+".pkl"
    if noclobber and os.path.isfile(outfilename):
        return

    atm_str = '--atm' if atm else ''
    telescope_str = '--telescope' if telescope else ''
    sensor_str = '--sensor' if sensor else ''
    filter_str = '-'+filter_name
    chip_str = '-'+chip

    subdir = '/nfs/slac/g/ki/ki19/jmeyers3/chroma/bin/phosim/validate/monochromatic_seeing/'
    command = '"cd {} && python measure_moffat_fwhm.py {} {} {} --wavelength {} {} {}"'
    command = command.format(subdir, atm_str, telescope_str, sensor_str, wavelength,
                             filter_str, chip_str)
    stdout_file = subdir+'stdout/'+'fitPSF_'+obshistid+chip
    job_name = obshistid+chip
    full_command = 'bsub -q medium -oo {} -J {} {}'.format(stdout_file, job_name, command)
    print full_command
    subprocess.call(full_command, shell=True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    for s in 'ugrizY147': # filters + chip #s
        parser.add_argument('-'+s, action='store_true')
    parser.add_argument('--atm', action='store_true')
    parser.add_argument('--telescope', action='store_true')
    parser.add_argument('--sensor', action='store_true')
    parser.add_argument('--wavelength', type=int)
    parser.add_argument('--submitall', action='store_true')
    parser.add_argument('--noclobber', action='store_true')
    parser.add_argument('--doall', action='store_true')
    args = parser.parse_args()

    waves = {'u': np.arange(325, 401, 25),
             'g': np.arange(400, 551, 25),
             'r': np.arange(550, 701, 25),
             'i': np.arange(675, 826, 25),
             'z': np.arange(800, 951, 25),
             'Y': np.arange(900, 1051, 25)}
    if args.submitall:
        for sensor in [0, 1]:
            for telescope in [0, 1]:
                for atm in [1]:
                    for f, ws in waves.iteritems():
                        for w in ws:
                            for c in '147':
                                submit_job(atm, telescope, sensor, w, f, c, 1000, args.noclobber)
    elif args.doall:
        for sensor in [0, 1]:
            for telescope in [0, 1]:
                for atm in [1]:
                    for f, ws in waves.iteritems():
                        for w in ws:
                            for c in [1,4,7]:
                                measure_moffat_fwhm(atm, telescope, sensor,
                                                    w, f, c, 1000, args.noclobber)
    else:
        for f in 'ugrizY':
            if not vars(args)[f]:
                continue
            for c in [1,4,7]:
                if not vars(args)[str(c)]:
                    continue
                measure_moffat_fwhm(args.atm, args.telescope, args.sensor,
                                    args.wavelength, f, c, 1000)
