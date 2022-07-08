# Plot LSST filter throughputs.

import os

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import galsim

import _mypath
import chroma

datadir = "../../data/"

fontsize = 13


def LSST_filters():
    waves = np.arange(300, 1101, 1, dtype=np.float64)
    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Wavelength (nm)", fontsize=fontsize)
    ax.set_xlim(300, 1100)
    ax.set_ylabel("Throughput", fontsize=fontsize)
    ax.set_ylim(0.0, 0.6)

    # 350nm wide Euclid filter.
    # ax.fill_between([0., 550., 550., 900., 900., 1200.], [-1, -1, 0.25, 0.25, -1, -1], -1,
    #                 color='black', alpha=0.15)

    colors = ["purple", "blue", "green", "gold", "magenta", "red"]
    for color, filter_ in zip(colors, "ugrizy"):
        # filters are stored in two columns: wavelength (nm), and throughput
        fdata = galsim.Bandpass(datadir + "filters/LSST_{}.dat".format(filter_), u.nm)
        fwave, throughput = fdata.wave_list, fdata(fdata.wave_list)
        ax.fill_between(fwave, throughput, 0.0, color=color, alpha=0.3)

    label_pos = {"u": 350.0, "g": 460, "r": 618, "i": 750, "z": 867, "y": 967}
    for k, v in label_pos.items():
        ax.text(v, 0.1, k, fontsize=fontsize)

    for label in ax.get_xticklabels():
        label.set_fontsize(fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(fontsize)

    if not os.path.isdir("output/"):
        os.mkdir("output/")
    fig.tight_layout()
    fig.savefig("output/LSST_filters.png", dpi=220)


if __name__ == "__main__":
    LSST_filters()
