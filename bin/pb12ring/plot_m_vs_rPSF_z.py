import numpy
import matplotlib.pyplot as plt

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

def main():
    infile = 'output/m_vs_rPSF_z.dat'
    calib = {'z':numpy.empty(0), 'r2':numpy.empty(0),
             'c1':numpy.empty(0), 'c2':numpy.empty(0),
             'm1':numpy.empty(0), 'm2':numpy.empty(0),
             'm_analytic':numpy.empty(0), 'c_analytic':numpy.empty(0)}
    try:
        with open(infile, 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = line.replace(':', ' ')
                line = ' '.join(line.split())
                z, r2, c1, c2, m1, m2, m_analytic = line.split(' ')
                calib['z'] = numpy.append(calib['z'], float(z))
                calib['r2'] = numpy.append(calib['r2'], float(r2))
                calib['c1'] = numpy.append(calib['c1'], float(c1))
                calib['c2'] = numpy.append(calib['c2'], float(c1))
                calib['m1'] = numpy.append(calib['m1'], float(c1))
                calib['m2'] = numpy.append(calib['m2'], float(c1))
                calib['m_analytic'] = numpy.append(calib['m_analytic'], float(m_analytic))
                calib['c_analytic'] = numpy.append(calib['m_analytic'], float(m_analytic) / 2.0)
    except IOError:
        pass
    uniqz = numpy.array(uniqify(calib['z']))
    uniqr2 = numpy.array(uniqify(calib['r2']))
    m1array = numpy.empty([len(uniqr2), len(uniqz)], dtype=numpy.float64)
    for iz, z in enumerate(uniqz):
        for ir2, r2 in enumerate(uniqr2):
            m1array[ir2, iz] = calib['m1'][numpy.logical_and(calib['z'] == z,
                                                             calib['r2'] == r2)]

    plt.imshow(m1array, extent=[uniqz.min(), uniqz.max(), uniqr2.min(), uniqr2.max()], aspect='auto')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
