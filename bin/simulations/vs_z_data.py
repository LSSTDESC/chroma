import subprocess

# # DCR only
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_GG_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_SG_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_GM_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_SM_DCR.dat"
# subprocess.call(cmd, shell=True)

# # Chromatic Seeing only
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_GG_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_SG_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_GM_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_SM_CS.dat"
# subprocess.call(cmd, shell=True)

# # Both
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_GG_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_SG_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_GM_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_SM_both.dat"
# subprocess.call(cmd, shell=True)

# # And now do it all again, but specify FWHM instead of r2:
# # DCR only
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_FWHM 0.7 --gal_convFWHM 0.868 --outfile output/FWHMfix_GG_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --outfile output/FWHMfix_SG_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_FWHM 0.7 --gal_convFWHM 0.868 --moffat --outfile output/FWHMfix_GM_DCR.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --alpha 0.0 --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --moffat --outfile output/FWHMfix_SM_DCR.dat"
# subprocess.call(cmd, shell=True)

# # Chromatic Seeing only
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_FWHM 0.7 --gal_convFWHM 0.868 --outfile output/FWHMfix_GG_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --outfile output/FWHMfix_SG_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_FWHM 0.7 --gal_convFWHM 0.868 --moffat --outfile output/FWHMfix_GM_CS.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --noDCR --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --moffat --outfile output/FWHMfix_SM_CS.dat"
# subprocess.call(cmd, shell=True)

# # Both
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_FWHM 0.7 --gal_convFWHM 0.868 --outfile output/FWHMfix_GG_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --outfile output/FWHMfix_SG_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_FWHM 0.7 --gal_convFWHM 0.868 --moffat --outfile output/FWHMfix_GM_both.dat"
# subprocess.call(cmd, shell=True)
# cmd = "python ring_vs_z.py --stamp_size 21 --PSF_FWHM 0.7 --gal_convFWHM 0.868 -n 4.0 --moffat --outfile output/FWHMfix_SM_both.dat"
# subprocess.call(cmd, shell=True)

# Try out the perturbative correction.
# DCR only
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_perturb_GG_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_perturb_SG_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_perturb_GM_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_perturb_SM_DCR.dat"
subprocess.call(cmd, shell=True)

# Chromatic Seeing only
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_perturb_GG_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_perturb_SG_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_perturb_GM_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_perturb_SM_CS.dat"
subprocess.call(cmd, shell=True)

# Both
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 --outfile output/r2fix_perturb_GG_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/r2fix_perturb_SG_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/r2fix_perturb_GM_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --perturb --stamp_size 21 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/r2fix_perturb_SM_both.dat"
subprocess.call(cmd, shell=True)
