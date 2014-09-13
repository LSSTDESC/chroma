import subprocess

bands = ['r', 'i']
types = ['Rbar', 'V', 'S_m02']
corrs = [('', 'few'),
         ('--corrected','corrected')]

for band in bands:
    for typ in types:
        for corr in corrs:
            cmd = "python plot_new_bias.py"
            cmd += ' '+typ
            cmd += " --band LSST_{}".format(band)
            cmd += ' '+corr[0]
            cmd += " --outfile output/d{}_{}_LSST_{}.png".format(typ, corr[1], band)
            print cmd
            subprocess.call(cmd, shell=True)
