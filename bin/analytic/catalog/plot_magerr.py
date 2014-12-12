import numpy as np
import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_magerr(args):
    stardata = cPickle.load(open(args.starfile))
    galdata = cPickle.load(open(args.galfile))

    x = np.arange(len(stardata))
    np.random.shuffle(x)
    stardata=stardata[x[:1000]]
    x = np.arange(len(galdata))
    np.random.shuffle(x)
    galdata=galdata[x[:1000]]

    xranges = {'u': (20, 28),
               'g': (20, 28),
               'r': (20, 28),
               'i': (20, 26),
               'z': (20, 26),
               'y': (20, 26)}

    fig = plt.figure(figsize=(8, 8))
    for i, f in enumerate('ugrizy'):
        ax = fig.add_subplot(3,2,i+1)
        ax.scatter(galdata['magCalc']['LSST_'+f], galdata['magErr']['LSST_'+f],
                   s=5, alpha=0.4, c='r')
        # ax.scatter(stardata['magCalc']['LSST_'+f], stardata['magErr']['LSST_'+f],
        #            s=2, alpha=0.5, c='b')
        ax.set_xlabel('mag')
        ax.set_ylabel('mag err')
        ax.set_ylim(0,0.2)
        ax.set_xlim(xranges[f])
        ax.text(0.1, 0.8, f, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig("output/magerr.pdf", dpi=220)

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--galfile", type=str, default="output/galaxy_data.pkl")
    parser.add_argument("--starfile", type=str, default="output/star_data.pkl")
    args = parser.parse_args()

    plot_magerr(args)
