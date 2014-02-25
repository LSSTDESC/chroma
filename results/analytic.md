---
layout: results
title: Analytic Results
next_section: catalog
permalink: /results/analytic/
---

The first script one should run when using Chroma is `bin/analytic/analytic_table.py`.  This script
computes synthetic photometry magnitudes and chromatic biases for a set of standard template stellar
and galactic SEDs, and stores the results for future use.

{% highlight bash %}
~ $ cd bin/analytic
~ $ python analytic_table.py
{% endhighlight %}

With these results stored, we can take our first look at some analytic predictions for chromatic
biases by running the following other scripts in the bin/analytic/ directory:

{% highlight bash %}
~ $ python plot_DCR_moment_shifts.py
~ $ python plot_PSF_size_shift.py
{% endhighlight %}

These will produce plots in the newly created bin/analytic/output/ directory:
{% highlight bash %}
~ $ ls
Rbar.LSST_g.png  Rbar.LSST_r.png   S.LSST_i.png
V.LSST_g.png     V.LSST_r.png      stars.pkl
Rbar.LSST_i.png  S.Euclid_350.png  S.LSST_r.png
V.LSST_i.png     galaxies.pkl
{% endhighlight %}

The `Rbar*png` figures show the centroid shifts due to DCR computed for different spectra relative to
the centroid shift of a G5V spectrum.

Similarly, the `V*png` figures show the DCR second moment shifts relative to a G5V star.

Finally, the `S*png` figures show the shift in PSF size, quantified as
\\(\\Delta r^2_{PSF}/r^2_{PSF}\\) due to chromatic seeing for the LSST filters, and due to a
Euclid-like \\(FWHM \\propto \\lambda^{+0.6}\\) relationship for the Euclid_350 filter.
