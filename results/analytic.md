---
layout: results
title: Analytic Results
next_section: ColorCorrect
permalink: /results/analytic/
---

The first script one should run when using Chroma is `bin/analytic/analytic_table.py`.  This script
computes synthetic photometry magnitudes and chromatic biases for a set of standard template stellar
and galactic SEDs, and stores the results for future use.

{% highlight bash %}
$ cd CHROMA_DIR/bin/analytic/
$ python analytic_table.py
{% endhighlight %}

With these results stored, we can take our first look at some analytic predictions for chromatic
biases by running the following scripts in the bin/analytic/ directory:

{% highlight bash %}
$ python plot_DCR_moment_shifts.py
$ python plot_PSF_size_shift.py
{% endhighlight %}

These will produce plots in the newly created bin/analytic/output/ directory:
{% highlight bash %}
$ ls output/*.png
output/Rbar.LSST_g.png        output/Rbar.LSST_r.png
output/S_m02.LSST_i.png       output/V.LSST_g.png
output/V.LSST_r.png           output/Rbar.LSST_i.png
output/S_p06.Euclid_350.png   output/S_m02.LSST_r.png
output/V.LSST_i.png
{% endhighlight %}

The `Rbar*png` figures show the centroid shifts due to DCR computed for different spectra relative
to the centroid shift of a G5V spectrum. Here's the plot for the LSST _r_-band.

<img src="{{site.url}}/img/Rbar.LSST_r.png" width="650">

Stellar spectra are represented by star symbols at redshift 0, while galaxy spectra are
represented by lines.  The wider and narrower gray bands show the requirements for DES and LSST
respectively.

Similarly, the `V*png` figures show the DCR second moment shifts relative to a G5V star.

<img src="{{site.url}}/img/V.LSST_r.png" width="650">

Finally, the `S*png` figures show the shift in PSF size, quantified as
\\(\\Delta r^2_{PSF}/r^2_{PSF}\\), due to chromatic seeing for the LSST filters, and due to a
Euclid-like \\(FWHM \\propto \\lambda^{+0.6}\\) relationship for the Euclid_350 filter.

<img src="{{site.url}}/img/S_m02.LSST_r.png" width="650">

<img src="{{site.url}}/img/S_p06.Euclid_350.png" width="650">

Clearly the biases on PSF shape and size due to chromatic effects exceed the requirements for
these surveys.