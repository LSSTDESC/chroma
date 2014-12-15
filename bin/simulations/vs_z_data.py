import subprocess
import os
from argparse import ArgumentParser

def vs_z_data(args):

    correction_modes = [("", "noCorr"),             # no correction
                        ("--perturb", "Perturb")]   # perturbative correction

    size_modes = [("--PSF_r2 0.42 --gal_r2 0.30", "r2r2"),              # hold r2_psf and r2_gal fixed
                  ("--PSF_FWHM 0.7 --gal_HLR 0.25", "FWHMHLR"),         # hold FWHM_psf and HLR_gal fixed
                  ("--PSF_FWHM 0.7 --gal_convFWHM 0.898", "FWHMFWHM")]  # hold FWHM_psf and (psf convolved with gal)_FWHM fixed

    physics_modes = [("", "both"),            # don't turn anything off
                     ("--alpha 0.0", "DCR"),  # turn off chromatic seeing,
                     ("--noDCR", "CS")]       # turn off DCR

    profile_modes = [("", "GG"),                 # Gaussian gal, Gaussian PSF
                     ("-n 4.0", "DG"),           # DeV galaxy, Gaussian PSF
                     ("--moffat", "GM"),         # Gaussian galaxy, Moffat PSF
                     ("-n 4.0 --moffat", "DM")]  # DeV galaxy, Moffat PSF

    stamp_size = 31

    for size_mode in size_modes:
        for physics_mode in physics_modes:
            for profile_mode in profile_modes:
                for correction_mode in correction_modes:
                    if not os.path.isdir("output/"):
                        os.mkdir("output")
                    outfilename = "output/ring_vs_z_"
                    outfilename += profile_mode[1]+'_'
                    outfilename += physics_mode[1]+'_'
                    outfilename += correction_mode[1]+'_'
                    outfilename += size_mode[1]+'.dat'
                    if not args.clobber and os.path.isfile(outfilename):
                        continue
                    cmd = "python ring_vs_z.py --stamp_size {}".format(stamp_size)
                    cmd += ' '+profile_mode[0]
                    cmd += ' '+physics_mode[0]
                    cmd += ' '+correction_mode[0]
                    cmd += ' '+size_mode[0]
                    cmd += " --outfile {}".format(outfilename)
                    print cmd
                    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--clobber', action='store_true')
    args = parser.parse_args()
    vs_z_data(args)
