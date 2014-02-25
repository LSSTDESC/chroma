---
layout: text
title: Getting Started
permalink: /GettingStarted/home/
---

Chroma is written entirely in Python.  Once downloaded, there is nothing to install, all python
scripts are intended to run from the directory in which they are unpacked.

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

Basic analytic results can be computed using the SEDs and filters included in data/SEDs/ and data/filters/.