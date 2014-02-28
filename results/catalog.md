---
layout: results
title: Catalog Results
prev_section: "ColorCorrect"
next_section: "MachineLearning"
permalink: /results/catalog/
---

The previous pages looked at chromatic biases that arise for a set of standard template stellar and
galactic SEDs.  In the real Universe, the distribution of SEDs is much more complicated. Here we
take a look at the distribution of chromatic biases for a more realistic set of SEDs obtained
from the LSST catalog framework.

We provide two methods for users to obtain a realistic catalog of stellar and galactic magnitudes
and chromatic biases.  The software to produce such a catalog using the LSST Image Simulation
catalog framework is included in Chroma, and its use is detailed below.  The catalog framework can
be a bit tricky to work with, however, and requires downloading a large collection (~3.5 Gb) of
unreddened rest-frame SEDs.  As an alternative, we also provide the resulting catalogs of magnitudes
and biases as a separate [download](http://slac.stanford.edu/~jmeyers3/).

### Making the chromatic bias catalogs

The first catalog script we'll look at is `make_catalogs.py`.  This is the only script in Chroma that
requires the LSST catalog framework to run.  It queries the LSST catalog simulation (CatSim) database
for stars and galaxies inside a 1 degree patch.  The stars are selected to have _i_ band magnitude
between 16 and 22, which is approximately the range useable for PSF estimation. The galaxies are
selected to be brighter than _i_ < 25.3, which is referred to as the LSST "gold sample" for weak
lensing.  These are galaxies that should yield reasonable signal-to-noise for shape estimates. The
returned catalog entries are written to the files `output/galaxy_catalog.dat` and
`output/star_catalog.dat`.

{% highlight bash %}
$ cd CHROMA_DIR/bin/analytic/catalog/
$ python make_catalogs.py
$ ls output/
output/galaxy_catalog.dat
output/star_catalog.dat
{% endhighlight %}

Let's take a look at the galaxy catalog file

{% highlight bash %}
$ head -n 3 output/galaxy_catalog.dat
galtileid, objectId, raJ2000, decJ2000, redshift, u_ab, g_ab, r_ab, i_ab, z_ab, y_ab, sedPathBulge, sedPathDisk, sedPathAgn, magNormBulge, magNormDisk, magNormAgn, internalAvBulge, internalRvBulge, internalAvDisk, internalRvDisk
222403644576, 222403644576, 199.44443756, -10.60410519, 0.61963493, 26.20507431, 26.12331772, 25.60255432, 25.20387077, 25.12633514, 25.05166245, None, galaxySED/Const.20E09.02Z.spec.gz, None, nan, 25.57798004, nan, 0.00000000, 3.09999990, 0.10000000, 3.09999990
222402644496, 222402644496, 199.44398768, -10.60823399, 0.41881949, 28.69147873, 26.64548302, 25.21138763, 24.58403778, 24.24159241, 24.04888725, None, galaxySED/Inst.50E09.02Z.spec.gz, None, nan, 24.80845070, nan, 0.00000000, 3.09999990, 0.30000001, 3.09999990
{% endhighlight %}

The first line shows the names of each database column requested in `make_catalogs.py`, and each subsequent line holds one the entries for one galaxy.  The columns are:

- galtileid, objectID : These keep track of which galaxy we're looking at
- raJ2000, decJ2000, redshift : Where the galaxy is located (in 3D)
- u_ab - y_ab : AB magnitudes through the LSST ugrizy filters.
- sedPathBulge : Name of the rest-frame unreddened SED file representing the bulge component of the galaxy.
- magNormBulge : The AB magnitude of the bulge component at 500nm rest-frame
- internalAvBulge, internalRvBulge : The rest frame reddening of the bulge component
- similar params for the disk component and possible AGN component

The stellar catalog has similar entries, although somewhat simpler since the stars are single
component.

Our goal is to take these parameters, construct the observed frame SED (including bulge, disk, and
AGN components, dust, and redshift), and calculate the relative chromatic biases for each star and
galaxy.  This is precisely what the scripts `process_gal_catalog.py` and `process_star_catalog.py`
do.  Users may in particular wish to look at the functions
`process_gal_catalog.py:composite_spectrum()` and `process_star_catalog.py:stellar_spectrum()` to see
exactly how the observed frame SEDs are generated.  The output of the processing scripts are pickle
files containing catalog and synthetic photometry, and calculated chromatic biases.

{% highlight bash %}
$ python process_gal_catalog.py
$ python process_star_catalog.py
$ ls output/*.pkl
output/galaxy_data.pkl
output/star_data.pkl
{% endhighlight %}

### Download chromatic biases catalogs

Since the LSST catalog framework can be somewhat tricky to install, we have also made the catalog
files that would be created by `process_gal_catalog.py` and `process_star_catalog.py` available
[online](http://slac.stanford.edu/~jmeyers3/).  You can download the files with a web browser and
place them in the bin/analytic/catalog/output/ directory or use a utility like curl:

{% highlight bash %}
$ cd DIRECTORY_TO_CHROMA/bin/analytic/catalog/output/
$ ls
$ curl -O slac.stanford.edu/~jmeyers3/galaxy_data.pkl
$ curl -O slac.stanford.edu/~jmeyers3/star_data.pkl
$ ls
galaxy_data.pkl
star_data.pkl
{% endhighlight %}

### Plots

With the processed catalog data in place, we can look at the distribution of chromatic biases for
a realistic population of lensing galaxies.  The tool for generating catalog bias plots is
`plot_bias.py`.  This script takes a number of command-line options, which can be listed with the
`--help` option:

{% highlight bash %}
$ python plot_bias.py --help
usage: plot_bias.py [-h] [--galfile GALFILE] [--starfile STARFILE]
                    [--corrected] [--band [BAND]] [--color COLOR COLOR]
                    [--outfile [OUTFILE]] [--nominal_plots]
                    [bias]

positional arguments:
  bias                 which chromatic bias to plot (Default: 'Rbar') Other
                       possibilities include: 'V', 'S_m02', 'S_p06', 'S_p10'

optional arguments:
  -h, --help           show this help message and exit
  --galfile GALFILE    input galaxy file. Default
                       'output/corrected_galaxy_data.pkl'
  --starfile STARFILE  input star file. Default
                       'output/corrected_star_data.pkl'
  --corrected          plot learning residuals instead of G5v residuals.
  --band [BAND]        band of chromatic bias to plot (Default: 'LSST_r')
  --color COLOR COLOR  color to use for symbol color (Default: ['LSST_r',
                       'LSST_i'])
  --outfile [OUTFILE]  output filename (Default: 'output/chromatic_bias.png')
  --nominal_plots      Plot some nominal useful LSST and Euclid figures
{% endhighlight %}

Notice that the default input files: `output/corrected_galaxy_data.pkl` and
`output/corrected_star_data.pkl` don't exist yet.  This is because these are the files that will
be created under the machine learning corrections step on the next page.  For now we can use the
uncorrected catalog files by adding them manually as command line arguments.  For instance, to make
a plot of the centroid shift bias in the _r_ band due to DCR, use:

{% highlight bash %}
$ python plot_bias.py Rbar --galfile output/galaxy_data.pkl \
                           --starfile output/star_data.pkl  \
                           --outfile output/dRbar.LSST_r.png
{% endhighlight %}

<img src="{{site.url}}/img/dRbar.LSST_r.png" width="650">

Each point of the scatter plot is one galaxy from the catalog.  The points are colored by their
_r_ - _i_ color, which as we showed in the previous page is correlated with the centroid shift.  The
histograms on the left show the distributions of centroid shifts for stars in blue, and the galaxies
projected along the redshift axis in red.  The dark horizontal bars again show the requirements for
DES and LSST.

To quickly plot a number of interesting plots, we can use `plot_bias.py` with the option
 `--nominal_plots`.
{% highlight bash %}
$ python plot_bias.py --galfile output/galaxy_data.pkl \
                      --starfile output/star_data.pkl \
                      --nominal_plots
{% endhighlight %}

Here are a few of the more interesting cases.

<img src="{{site.url}}/img/dV.LSST_r.png" width="650">
<img src="{{site.url}}/img/dS_m02.LSST_r.png" width="650">
<img src="{{site.url}}/img/dS_p06.Euclid_350.png" width="650">

The last plot here, for the Euclid telescope, is slightly misleading since we generated a catalog
appropriate for LSST, and Euclid will be a much shallower survey.  However, the peak of the
Euclid chromatic PSF bias is around redshift 1.0, where Euclid will still be quite sensitive.  Also
notice that the symbol colors for this plot represent *LSST* _r_ - _i_ color.  The proposed Euclid
footprint will overlap the LSST footprint by about 5000 square degrees.

Notice also that while chromatic biases are clearly correlated with _r_ - _i_, they aren't perfectly
correlated.  That is, for a given chromatic bias, or horizontal cut across the above scatter plots,
more than one _r_ - _i_ color, or symbol color, occurs.