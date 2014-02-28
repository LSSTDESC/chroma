---
layout: results
title: Color Corrections
prev_section: analytic
next_section: catalog
permalink: /results/ColorCorrect/
---

Chromatic biases depend on the slope of the SED across the filter being imaged through.  As such,
we might expect the biases to be correlated with the photometric colors of nearby bands.  The
remaining Python scripts in the `bin/analytic/` directory investigate this possibility:

{% highlight bash %}
$ cd CHROMA_DIR/bin/analytic/
$ python plot_DCR_color_corrections.py
$ python plot_PSF_size_color_corrections.py
$ ls output/*vs*png
output/Rbar_LSST_r_vs_LSST_r-LSST_i.png
output/V_LSST_r_vs_LSST_r-LSST_i.png
output/S_p06_Euclid_350_vs_LSST_r-LSST_i.png
output/S_m02_LSST_r_vs_LSST_r-LSST_i.png
{% endhighlight %}

Let's look at the DCR centroid shift plot:

<img src="{{site.url}}/img/Rbar_LSST_r_vs_LSST_r-LSST_i.png" width="650">

In the top panel, the same chromatic biases as on the previous page are being plotted on the
y-axis, but the x-axis has been replaced with the LSST _r_ - _i_ color.  Clearly there is a
correlation.

Let's look at the DCR second moment shifts now:

<img src="{{site.url}}/img/V_LSST_r_vs_LSST_r-LSST_i.png" width="650">

The correlation here is much weaker, and the residuals are not small enough to meet LSST
requirements.  Presumably, if we used the information in all six photometric bands we could do
better still.  We will investigate this possibility shortly.
