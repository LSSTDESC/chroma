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

- skikit-learn

one can also use machine learning algorithms to predict the size of chromatic biases given
photometric data, which can then lead to a correction mitigating the bias.

Finally, to test analytic results in simulations, the following packages are also required:

- GalSim
- lmfit
- either astropy or pyfits

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

We'll briefly go through the three types of analysis covered in these directories here.

### Analytic Results

Basic analytic results can be computed using the SEDs and filters included in data/SEDs/
and data/filters/.