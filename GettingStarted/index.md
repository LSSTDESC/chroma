---
layout: text
title: Getting Started
permalink: /GettingStarted/home/
---

Dependencies
------------

Chroma is written entirely in Python.  Once downloaded, there is nothing to install; all python
scripts are intended to run from the directory in which they are unpacked.

Chroma requires several python libraries to run, although not all libraries are required for all
analyses.  The basic set of dependencies, required everywhere, are

- numpy
- scipy
- matplotlib

which are sufficient to enable one to evaluate chromatic biases and make plots.

With the added dependency

- scikit-learn

one can also use machine learning algorithms to predict the size of chromatic biases given
photometric data, which can then lead to a correction mitigating the bias.  This step requires
a catalog on which to run, which can either be built using the LSST catalogs framework with
the script supplied, or alternatively can simply be downloaded
[here](http://slac.stanford.edu/~jmeyers3/).

Finally, to test analytic results in simulated images, the following packages are required:

- GalSim
- lmfit
- either astropy or pyfits

Actually, astropy is a nice tool to have around for the earlier analyses too, since, when available,
we use its console module to create a nice looking progress bar for steps that take a while.

Directory Structure
-------------------

The primary directory structure of chroma is the following:

{% highlight yaml %}
   - bin/
     - bin/analytic/
       - bin/analytic/machine_learning/
     - bin/intuition/
     - bin/simulations/
   - chroma/
   - data/
     - data/SEDs/
     - data/filters/
{% endhighlight %}

(There are additional directories lurking, but the above are the ones that are most interesting and
up-to-date).

The data/ directory contains files representing a small number of SEDs, and the LSST filter
bandpasses.

The chroma/ directory contains a few useful library modules:

- dcr.py -  Code to convert wavelength and zenith angle into a refraction angle.
- galtool.py - Code to use GalSim and lmfit jointly, i.e. translate lmfit Parameters describing a
               galaxy into an actual image.
- utils.py - A few utility functions.
- plot.py - Plotting routines.
- sed.py - Code that extends the galsim.SED class to analytically compute magnitudes and chromatic
           biases.
- sampled.py - Code that implements simple SED and Bandpass classes defined by samples in
               wavelength.

Scripts that produce output reside in the bin/ directory.  We will go through the various bin/
scripts under the results tab above.