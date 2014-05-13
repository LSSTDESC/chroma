import subprocess
import os

correction_modes = [("", "noCorr"),             # no correction
                    ("--perturb", "Perturb")]   # perturbative correction

size_modes = [("--PSF_FWHM 0.7 --gal_HLR 0.225", "FWHMHLR"),       # hold FWHM_psf and HLR_gal fixed
              ("--PSF_r2 0.42 --gal_r2 0.27", "r2r2"),             # hold r2_psf and r2_gal fixed
              ("--PSF_FWHM 0.7 --gal_convFWHM 0.86", "FWHMFWHM")]  # hold FWHM_psf and (psf convolved with gal)_FWHM fixed

physics_modes = [("", "both"),            # don"t turn anything off
                 ("--alpha 0.0", "DCR"),  # turn off chromatic seeing,
                 ("--noDCR", "CS")]       # turn off DCR

profile_modes = [("", "GG"),                 # Gaussian gal, Gaussian PSF
                 ("-n 4.0", "DG"),           # DeV galaxy, Gaussian PSF
                 ("--moffat", "GM"),         # Gaussian galaxy, Moffat PSF
                 ("-n 4.0 --moffat", "DM")]  # DeV galaxy, Moffat PSF

stamp_size = 21

for size_mode in size_modes:
    for physics_mode in physics_modes:
        for profile_mode in profile_modes:
            for correction_mode in correction_modes:
                cmd = "python ring_vs_z.py --stamp_size {}".format(stamp_size)
                cmd += ' '+profile_mode[0]
                cmd += ' '+physics_mode[0]
                cmd += ' '+correction_mode[0]
                cmd += ' '+size_mode[0]
                outfilename = "output/ring_vs_z_"
                outfilename += profile_mode[1]+'_'
                outfilename += physics_mode[1]+'_'
                outfilename += correction_mode[1]+'_'
                outfilename += size_mode[1]+'.dat'
                cmd += " --outfile {}".format(outfilename)
                if os.path.isfile(outfilename):
                    continue
                print cmd
                subprocess.call(cmd, shell=True)
