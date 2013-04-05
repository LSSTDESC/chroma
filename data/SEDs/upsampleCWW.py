import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("CWW*old.ascii")

for f in files:
    data = np.genfromtxt(f)
    wave = data[:,0]
    flambda = data[:,1]
    waveout = np.arange(wave.min(), wave.max(), 0.5)
    flambda_i = np.interp(waveout, wave, flambda)

    with open(f.replace("ext_old.ascii", "ext.ascii"), "w") as fil:
        for w0, f0 in zip(waveout, flambda_i):
            fil.write("{} {}\n".format(w0, f0))
