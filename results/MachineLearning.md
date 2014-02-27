---
layout: results
title: Machine Learning
prev_section: "catalog"
next_section: "ModelFittingBias"
permalink: /results/MachineLearning/
---

We showed earlier that a single color measurement can be used to help predict the magnitude of a
chromatic bias.  One step beyond this color correction is to predict chromatic biases using the
photometry of all 6 LSST filters, which we show on this page.

The scripts `gal_ML.py` and `star_ML.py` take the processed CatSim catalogs from the previous page
and, using a machine learning algorithm called Support Vector Regression, learn to predict chromatic
biases from LSST photometry.