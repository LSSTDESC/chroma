---
layout: text
title: Getting Started
permalink: /GettingStarted/home/
---

Dependencies
------------

Chroma is written entirely in Python.  Once downloaded, there is nothing to install, all python
scripts are intended to run from the directory in which they are unpacked.

Chroma requires several python libraries to run, although not all libraries are required for all
analyses.  The basic set of dependencies includes

- numpy
- scipy
- matplotlib

which will enable one to evaluate chromatic biases and make plots.

With the added dependency

- scikit-learn

one can also use machine learning algorithms to predict the size of chromatic biases given
photometric data, which can then lead to a correction mitigating the bias.  This step requires
a catalog on which to run, which can either be built using the LSST catalogs framework with
the script supplied, or alternatively can simply be downloaded from XXXXX.

Finally, to test analytic results in simulated images, the following packages are required:

- GalSim
- lmfit
- either astropy or pyfits

Actually, astropy is a nice tool to have around for earlier analyses too, since we use it's
console module to create a nice looking progress bar for steps that take a while when available.

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

The data/ directory contains files representing a small number of SEDs, and the LSST filter
bandpasses.

The chroma/ directory contains a few useful library modules:

- dcr.py: Code to convert wavelength and zenith angle into a refraction angle.
- galtool.py: Code to use GalSim and lmfit jointly, i.e. translate lmfit Parameters describing a
              galaxy into an actual image.
- utils.py: A few utility functions.
- plot.py: Plotting routines.

Scripts that produce output all reside in the bin/ directory.  We will go through the various
scripts under the results tab above.