# Average together indiv CCD filters obtained from 
# http://www.ioa.s.u-tokyo.ac.jp/~doi/sdss/SDSSresponse.html

import collections
import numpy as np

for f in 'ugriz':
    throughput = {}
    for i in range(6):
        wv, tp = np.genfromtxt("{}{}.dat".format(f, i+1)).T
        for w, t in zip(wv, tp):
            if w not in throughput:
                throughput[w] = (t, 1)
            else:
                throughput[w] = (throughput[w][0] + t, throughput[w][1]+1)
    file_ = open("SDSS_{}.dat".format(f), 'w')
    throughput = collections.OrderedDict(sorted(throughput.items()))
    for w, t in throughput.iteritems():
        file_.write("{:7.1f}  {:6.4f} \n".format(w, t[0]/t[1]))
    file_.close()
