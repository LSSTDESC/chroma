import _mypath
import PhosimObject
import numpy as np
import os

print 'TEST FILE!'
print 'phosim should(!) report the same fluxes for these... (maybe airmass dependent?)'
thisdir = os.path.dirname(__file__)
datadir = os.path.join(thisdir, '../data/')
filter_data = np.genfromtxt(datadir+'filters/LSST_r.dat')
filter_wave = filter_data[:,0]
filter_throughput = filter_data[:,1]
obj = PhosimObject.PhosimObject()
SEDs = ['PB12/'+i+'.ascii' for i in ['CWW_E_ext', 'CWW_Im_ext', 'CWW_Sbc_ext',
                                     'CWW_Scd_ext', 'KIN_SB1_ext', 'KIN_SB6_ext',
                                     'KIN_Sa_ext', 'KIN_Sb_ext', 'uko5v', 'ukb5iii',
                                     'uka5v', 'ukf5v', 'ukg5v', 'ukk5v', 'ukm5v']]
decs = np.linspace(-0.07, 0.07, len(SEDs))
RAs = np.linspace(-0.07, 0.07, len(SEDs))
zs = np.linspace(0.0, 0.8, len(SEDs))
for idec in range(len(decs)):
    for iRA in range(len(RAs)):
        obj.RA = RAs[iRA]
        obj.dec = decs[idec]
        obj._id = idec + iRA*0.01
        obj.redshift = zs[iRA]
        obj.SED_name = SEDs[idec]
        # Want 10^4 photons in filter specified by filter_wave and filter_throughput
        obj.set_flux_goal(1e4, filter_wave, filter_throughput)
        print obj.out_string()
