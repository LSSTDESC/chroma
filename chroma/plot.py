import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def chroma_fill_between(x, y1, y2=None, c=None, cm=cm.gist_rainbow, axes=None):
    """Plot y vs. x but fill in between y and 0 with a colormap.

    Arguments
    ---------
    x, y1, y2 -- curves to plot between
    c -- array congruent to x and y indicating which color in the
         color map to fill in at that point.  The range should be from
         0.0 to 1.0

    Keyword Arguments
    -----------------

    cm -- Matplotlib Colormap Object.  The c array above picks colors
          from this array. (default matplotlib.cm.gist_rainbow, which
          covers roygbiv with red at c=0.0 and violet at c=1.0)

    axes -- Matplotlib Axes object to use to make the plot.
    """

    if axes is None: axes = plt.gca()

    if y2 is None:
        y2 = np.zeros(len(y1), dtype=float)
    elif isinstance(y2, (int, float)):
        y2 *= np.ones(len(y1), dtype=float)

    cc = c.copy()
    cc[cc > 1.0] = 1.0
    cc[cc < 0.0] = 0.0

    for i in xrange(cm.N):
        cmin = 1.0*i/cm.N
        cmax = 1.0*(i+1)/cm.N
        w = np.int_(np.logical_and(cc >= cmin, cc <= cmax))
        if not w.any() : continue

        starts = np.nonzero(np.diff(w) == 1)[0]
        if w[0] == True :
            starts = np.insert(starts, 0, 0)

        ends = np.nonzero(np.diff(w) == -1)[0]
        if w[-1] == True :
            ends = np.append(ends, 0)

        for start, end in zip(starts, ends):
            axes.fill_between(x[start:end+1], y1[start:end+1], y2[start:end+1], color=cm(i))

def test():
    x = np.linspace(0.0, 2.0*np.pi, 1000)
    y = np.sin(x)

    # loop over colormap twice
    c = np.concatenate([np.linspace(0.0, 1.0, 500), np.linspace(0.0, 1.0, 500)])

    plt.plot(x, y, color="black")
    chroma_fill_between(x, y, c=c)
    plt.show()

if __name__ == "__main__":
    test()
