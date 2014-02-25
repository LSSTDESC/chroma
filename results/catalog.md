---
layout: results
title: Catalog Results
prev_section: "ColorCorrect"
next_section: "MachineLearning"
permalink: /results/catalog/
---

The previous pages looked at chromatic biases that arise for a set of standard template stellar and
galactic SEDs.  In the real Universe, the distribution of SEDs is much more complicated, of course.
Here we take a look at the distribution of chromatic biases for a more realistic set of SEDs obtained
from the LSST catalog framework.

### Making catalog data

The first catalog script we'll look at is `make_catalogs.py`.  This is the only script in Chroma that
requires the LSST catalog framework be installed to run.  It queries the LSST catalog simulation
(CatSim) database for stars and galaxies inside a 1 degree patch.  The stars are selected to have
_i_ band magnitude between 16 and 22, which is approximately the range useable for PSF estimation.
The galaxies are selected to be brighter than _i_ < 25.3, which is referred to as the LSST "gold
sample" for weak lensing.  The returned catalog entries are written to the files
`output/galaxy_catalog.dat` and `output/star_catalog.dat`.

{% highlight bash %}
$ cd CHROMA_DIR/bin/analytic/catalog/
$ python make_catalogs.py
$ ls output/
output/galaxy_catalog.dat
output/star_catalog.dat
{% endhighlight %}

Since the LSST catalog framework can be somewhat tricky to install, we have also made the catalog
files that would be created by `make_catalogs.py` available
[online](http://slac.stanford.edu/~jmeyers3/).  You can download the files with a web browser and
place them in the bin/analytic/catalog/output/ directory or use a utility like curl:

{% highlight bash %}
$ cd output/
$ ls
$ curl -O slac.stanford.edu/~jmeyers3/galaxy_catalog.dat
$ curl -O slac.stanford.edu/~jmeyers3/star_catalog.dat
$ ls
galaxy_catalog.dat
star_catalog.dat
$ cd ..
{% endhighlight %}

Let's take a look at the galaxy catalog file

{% highlight bash %}
$ head -n 3 output/galaxy_catalog.dat
galtileid, objectId, raJ2000, decJ2000, redshift, u_ab, g_ab, r_ab, i_ab, z_ab, y_ab, sedPathBulge, sedPathDisk, sedPathAgn, magNormBulge, magNormDisk, magNormAgn, internalAvBulge, internalRvBulge, internalAvDisk, internalRvDisk
222403644576, 222403644576, 199.44443756, -10.60410519, 0.61963493, 26.20507431, 26.12331772, 25.60255432, 25.20387077, 25.12633514, 25.05166245, None, galaxySED/Const.20E09.02Z.spec.gz, None, nan, 25.57798004, nan, 0.00000000, 3.09999990, 0.10000000, 3.09999990
222402644496, 222402644496, 199.44398768, -10.60823399, 0.41881949, 28.69147873, 26.64548302, 25.21138763, 24.58403778, 24.24159241, 24.04888725, None, galaxySED/Inst.50E09.02Z.spec.gz, None, nan, 24.80845070, nan, 0.00000000, 3.09999990, 0.30000001, 3.09999990
{% endhighlight %}

The first line shows names of each database column requested, and each subsequent line holds one
galaxy.  Most of the column names should be self-evident.  It is important to recognize that CatSim
represents galaxies as having multiple components, specifically a bulge, a disk, and potentially a
central AGN.

In order to evaluate chromatic biases, we need SEDs.  To create an SED from a catalog entry, we must
read a bulge SED filename ('sedPathBulge'), renormalize its magnitude ('magNormBulge'), apply dust
extinction ('internalAVBulge' and 'internalRVBulge'), redshift it ('redshift'), and add in similar
contributions from the disk and AGN components.  This is exactly what the `composite_spectrum`
function in the script, `process_gal_catalog.py` does.  Additionally, `process_gal_catalog.py`
integrates the SED over bandpasses to synthesize photometry and also uses the SEDs to compute
chromatic biases.  All of these computations are then stored in the file `output/galaxy_data.pkl`.
The situation for stars is similar, although the SEDs are somewhat simpler to define since they are
single component.

{% highlight bash %}
$ python process_gal_catalog.py
$ python process_star_catalog.py
$ ls output/*.pkl
output/galaxy_data.pkl
output/star_data.pkl
{% endhighlight %}

These scripts require that the unreddened, unredshifted SED files are available, of course.  These
can be downloaded using curl (~3.5G). Additionally, an environment variable needs to be set so that
the SEDs can be found:

{% highlight bash %}
$ cd DIRECTORY_FOR_SEDS
$ curl -O www.astro.washington.edu/users/krughoff/data/focal_plane_data.tar.gz
$ tar -xvzpf focal_plane_data.tar.gz
$ export $CAT_SHARE_DATA=DIRECTORY_FOR_SEDS
{% endhighlight %}

If you don't want to download all of the SEDs and process the catalogs yourself, then you can
download the output pkl files we've placed online.

{% highlight bash %}
$ cd CHROMA_DIR/bin/analytic/catalog/output
$ curl -O slac.stanford.edu/~jmeyers3/galaxy_data.pkl
$ curl -O slac.stanford.edu/~jmeyers3/star_data.pkl
$ cd ../
{% endhighlight %}

### Plots

With the processed catalog data in place, we can look at the distribution of chromatic biases for
a realistic population of lensing galaxies.  The tool for generating catalog bias plots is
`plot_bias.py`.  This script takes a number of command-line options, which can be listed with

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

At this point we haven't created `output/corrected_star_data.pkl` or
`output/corrected_gal_data.pkl`, so we'll have to override these with the --galfile and
--starfile options.  Let's make a plot of the centroid biases shift for the LSST _r_ band:

{% highlight bash %}
$ python plot_bias.py Rbar --galfile output/galaxy_data.pkl --starfile output/star_data.pkl --outfile output/dRbar.LSST_r.png
{% endhighlight %}

<img src="{{site.url}}/img/dRbar.LSST_r.png" width="650">

Each point of the scatter plot is one galaxy from the catalog.  The points are colored by their
_r_ - _i_ color, which as we showed in the previous page is correlated with the centroid shift.  The
histograms on the left show the distributions of centroid shifts for stars in blue, and the galaxies
projected across the redshift axis in red.  The dark horizontal bars again show the requirements for
DES and LSST.

To quickly plot a number of interesting plots, we can use `plot_bias.py` with the option
 `--nominal_plots`.
{% highlight bash %}
$ python plot_bias.py --galfile output/galaxy_data.pkl --starfile output/star_data.pkl --nominal_plots
{% endhighlight %}

Here are a few of the more interesting cases.

<img src="{{site.url}}/img/dV.LSST_r.png" width="650">
<img src="{{site.url}}/img/dS_m02.LSST_r.png" width="650">
<img src="{{site.url}}/img/dS_p06.Euclid_350.png" width="650">
