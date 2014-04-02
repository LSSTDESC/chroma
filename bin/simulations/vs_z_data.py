import subprocess

# DCR only
cmd = "python ring_vs_z.py --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --outfile output/GG_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/SG_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/GM_DCR.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --alpha 0.0 --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/SM_DCR.dat"
subprocess.call(cmd, shell=True)

# Chromatic Seeing only
cmd = "python ring_vs_z.py --noDCR --PSF_r2 0.42 --gal_r2 0.27 --outfile output/GG_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/SG_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --noDCR --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/GM_CS.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --noDCR --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/SM_CS.dat"
subprocess.call(cmd, shell=True)

# Both
cmd = "python ring_vs_z.py --PSF_r2 0.42 --gal_r2 0.27 --outfile output/GG_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --outfile output/SG_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --PSF_r2 0.42 --gal_r2 0.27 --moffat --outfile output/GM_both.dat"
subprocess.call(cmd, shell=True)
cmd = "python ring_vs_z.py --PSF_r2 0.42 --gal_r2 0.27 -n 4.0 --moffat --outfile output/SM_both.dat"
subprocess.call(cmd, shell=True)
