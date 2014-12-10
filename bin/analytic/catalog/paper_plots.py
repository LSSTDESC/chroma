import subprocess

bands = ['r', 'i']
types = ['LnRbarSqr', 'V', 'S_m02']
corrs = [('', 'few'),
         ('--corrected','corrected')]

for band in bands:
    for typ in types:
        for corr in corrs:
            cmd = "python plot_bias.py"
            cmd += " --galfile output/corrected_galaxy_data.pkl"
            cmd += ' '+typ
            cmd += " --band LSST_{}".format(band)
            cmd += ' '+corr[0]
            cmd += " --outfile output/d{}_{}_LSST_{}.pdf".format(typ, corr[1], band)
            print cmd
            subprocess.call(cmd, shell=True)
