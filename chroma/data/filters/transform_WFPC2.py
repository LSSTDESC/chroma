# rewrite WFPC2 filters using nanometers for wavelengths instead of Angstroms

import numpy

f814_data = numpy.genfromtxt('WFPC2_F814W.angstrom.dat')
f = open('WFPC2_F814W.dat', 'w')
for w, tp in f814_data:
    f.write('{:9.1f} {: 15.4E}\n'.format(w * 0.1, tp))


f606_data = numpy.genfromtxt('WFPC2_F606W.angstrom.dat')
f = open('WFPC2_F606W.dat', 'w')
for w, tp in f606_data:
    f.write('{:9.1f} {: 15.4E}\n'.format(w * 0.1, tp))
