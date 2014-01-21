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
    infile = 'output/m_vs_n_rPSF.dat'
    calib = {'r2':numpy.empty(0), 'n':numpy.empty(0),
             'c1':numpy.empty(0), 'c2':numpy.empty(0),
             'm1':numpy.empty(0), 'm2':numpy.empty(0)}
    try:
        with open(infile, 'r') as fil:
            for line in fil:
                line = line.replace('(', ' ')
                line = line.replace(')', ' ')
                line = line.replace(',', ' ')
                line = line.replace(':', ' ')
                line = ' '.join(line.split())
                r2, n, c1, c2, m1, m2 = line.split(' ')
                calib['r2'] = numpy.append(calib['r2'], float(numpy.sqrt(float(r2)) * 0.2))
                calib['n'] = numpy.append(calib['n'], float(n))
                calib['c1'] = numpy.append(calib['c1'], float(c1))
                calib['c2'] = numpy.append(calib['c2'], float(c1))
                calib['m1'] = numpy.append(calib['m1'], float(c1))
                calib['m2'] = numpy.append(calib['m2'], float(c1))
    except IOError:
        pass
    uniqr2 = numpy.array(uniqify(calib['r2']))
    uniqn = numpy.array(uniqify(calib['n']))
    m1array = numpy.empty([len(uniqr2), len(uniqn)], dtype=numpy.float64)
    for i_n, n in enumerate(uniqn):
        for ir2, r2 in enumerate(uniqr2):
            m1array[ir2, i_n] = calib['m1'][numpy.logical_and(calib['n'] == n,
                                                             calib['r2'] == r2)]

    plt.imshow(m1array, extent=[uniqn.min(), uniqn.max(), uniqr2.min(), uniqr2.max()], aspect='auto')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
