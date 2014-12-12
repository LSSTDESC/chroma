""" Use a machine learning algorithm, currently hard-coded as Support Vector Regression -- though
this is easily changeable, to learn chromatic biases as function of either colors, magnitudes, or
a combination of both.

Chromatic biases include:
  Rbar - centroid shift due to differential chromatic refraction.
  V - zenith-direction second moment shift due to differential chromatic refraction
  S - shift in "size" of the PSF due to a power-law dependence of the FWHM with wavelength:
      FWHM \propto \lambda^{\alpha}.  S = the second moment square radius r^2 = Ixx + Iyy.
      Three cases are tabulated:
        \alpha = -0.2 : appropriate for atmospheric chromatic seeing.  denoted 'S_m02'
        \alpha = 1.0 : appropriate for a pure diffraction limited PSF.  denoted 'S_p10'
        \alpha = 0.6 : appropriate for Euclid (see Voigt+12 or Cypriano+10).  denoted 'S_p06'
"""


import cPickle
from argparse import ArgumentParser

import numpy as np

def ML(train_objs, test_objs, predict_var=None, predict_band=None,
        use_color=False, use_mag=False, regressor="SVR"):

    if predict_var is None:
        raise ValueError
    ntrain = len(train_objs)
    ntest = len(test_objs)
    train_Y = np.empty(ntrain, dtype=np.float)
    test_Y = np.empty(ntest, dtype=np.float)
    if predict_band is None:
        train_Y[:] = train_objs[predict_var]
        test_Y[:] = test_objs[predict_var]
    else:
        train_Y[:] = train_objs[predict_var][predict_band]
        test_Y[:] = test_objs[predict_var][predict_band]

    if use_color: # use only five colors to train
        train_X = np.empty([ntrain, 5], dtype=np.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u'] - train_objs['magCalc']['LSST_g']
        train_X[:,1] = train_objs['magCalc']['LSST_g'] - train_objs['magCalc']['LSST_r']
        train_X[:,2] = train_objs['magCalc']['LSST_r'] - train_objs['magCalc']['LSST_i']
        train_X[:,3] = train_objs['magCalc']['LSST_i'] - train_objs['magCalc']['LSST_z']
        train_X[:,4] = train_objs['magCalc']['LSST_z'] - train_objs['magCalc']['LSST_y']

        # test data
        test_X = np.empty([ntest, 5], dtype=np.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u'] - test_objs['magCalc']['LSST_g']
        test_X[:,1] = test_objs['magCalc']['LSST_g'] - test_objs['magCalc']['LSST_r']
        test_X[:,2] = test_objs['magCalc']['LSST_r'] - test_objs['magCalc']['LSST_i']
        test_X[:,3] = test_objs['magCalc']['LSST_i'] - test_objs['magCalc']['LSST_z']
        test_X[:,4] = test_objs['magCalc']['LSST_z'] - test_objs['magCalc']['LSST_y']
    elif use_mag: # use only six magnitudes to train
        train_X = np.empty([ntrain, 6], dtype=np.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u']
        train_X[:,1] = train_objs['magCalc']['LSST_g']
        train_X[:,2] = train_objs['magCalc']['LSST_r']
        train_X[:,3] = train_objs['magCalc']['LSST_i']
        train_X[:,4] = train_objs['magCalc']['LSST_z']
        train_X[:,5] = train_objs['magCalc']['LSST_y']

        # test data
        test_X = np.empty([ntest, 6], dtype=np.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u']
        test_X[:,1] = test_objs['magCalc']['LSST_g']
        test_X[:,2] = test_objs['magCalc']['LSST_r']
        test_X[:,3] = test_objs['magCalc']['LSST_i']
        test_X[:,4] = test_objs['magCalc']['LSST_z']
        test_X[:,5] = test_objs['magCalc']['LSST_y']
    else: # default: use i-band magnitude, and five colors to train
        train_X = np.empty([ntrain, 6], dtype=np.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u'] - train_objs['magCalc']['LSST_g']
        train_X[:,1] = train_objs['magCalc']['LSST_g'] - train_objs['magCalc']['LSST_r']
        train_X[:,2] = train_objs['magCalc']['LSST_r'] - train_objs['magCalc']['LSST_i']
        train_X[:,3] = train_objs['magCalc']['LSST_i'] - train_objs['magCalc']['LSST_z']
        train_X[:,4] = train_objs['magCalc']['LSST_z'] - train_objs['magCalc']['LSST_y']
        train_X[:,5] = train_objs['magCalc']['LSST_i']

        # test data
        test_X = np.empty([ntest, 6], dtype=np.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u'] - test_objs['magCalc']['LSST_g']
        test_X[:,1] = test_objs['magCalc']['LSST_g'] - test_objs['magCalc']['LSST_r']
        test_X[:,2] = test_objs['magCalc']['LSST_r'] - test_objs['magCalc']['LSST_i']
        test_X[:,3] = test_objs['magCalc']['LSST_i'] - test_objs['magCalc']['LSST_z']
        test_X[:,4] = test_objs['magCalc']['LSST_z'] - test_objs['magCalc']['LSST_y']
        test_X[:,5] = test_objs['magCalc']['LSST_i']

    from sklearn.preprocessing import StandardScaler

    Y_scaler = StandardScaler().fit(train_Y)
    X_scaler = StandardScaler().fit(train_X)

    scaled_train_Y = Y_scaler.transform(train_Y)
    scaled_train_X = X_scaler.transform(train_X)
    scaled_test_X = X_scaler.transform(test_X)

    if regressor == 'SVR':
        from sklearn.svm import SVR
        learner = SVR(C=100, gamma=0.1)
    elif regressor == 'RandomForest':
        from sklearn.ensemble import RandomForestRegressor
        learner = RandomForestRegressor(200)
    elif regressor == 'ExtraTrees':
        from sklearn.ensemble import ExtraTreesRegressor
        learner = ExtraTreesRegressor(200)

    learner.fit(scaled_train_X, scaled_train_Y)
    predict_Y = Y_scaler.inverse_transform(learner.predict(scaled_test_X))
    return predict_Y

def gal_ML(train_objs, test_objs, **kwargs):

    # do some pruning of troublesome objects.  These generally fail when the SED
    # isn't blue enough to cover the wavelength range of the filter.
    good = np.ones(train_objs.shape, dtype=np.bool)
    for f in 'ugrizy':
        good = good & np.isfinite(train_objs['magCalc']['LSST_{}'.format(f)])
        good = good & np.isfinite(train_objs['magCalc']['LSST_{}'.format(f)])
    # kill everything with an AGN for now since mags don't match well.
    good = good & np.isnan(train_objs.magNormAGN)
    train_objs = train_objs[good]

    good = np.ones(test_objs.shape, dtype=np.bool)
    for f in 'ugrizy':
        good = good & np.isfinite(test_objs['magCalc']['LSST_{}'.format(f)])
        good = good & np.isfinite(test_objs['magCalc']['LSST_{}'.format(f)])
    # kill everything with an AGN for now since mags don't match well.
    good = good & np.isnan(test_objs.magNormAGN)
    test_objs = test_objs[good]

    ugrizy = [('LSST_u', np.float),
              ('LSST_g', np.float),
              ('LSST_r', np.float),
              ('LSST_i', np.float),
              ('LSST_z', np.float),
              ('LSST_y', np.float)]
    ugrizyE = [('LSST_u', np.float),
               ('LSST_g', np.float),
               ('LSST_r', np.float),
               ('LSST_i', np.float),
               ('LSST_z', np.float),
               ('LSST_y', np.float),
               ('Euclid_150', np.float),
               ('Euclid_250', np.float),
               ('Euclid_350', np.float),
               ('Euclid_450', np.float)]
    E = [('Euclid_150', np.float),
         ('Euclid_250', np.float),
         ('Euclid_350', np.float),
         ('Euclid_450', np.float)]

    data = np.recarray((len(test_objs),),
                          dtype = [('galTileID', np.uint64),
                                   ('objectID', np.uint64),
                                   ('raJ2000', np.float),
                                   ('decJ2000', np.float),
                                   ('redshift', np.float),
                                   ('sedPathBulge', np.str_, 64),
                                   ('sedPathDisk', np.str_, 64),
                                   ('sedPathAGN', np.str_, 64),
                                   ('magNormBulge', np.float),
                                   ('magNormDisk', np.float),
                                   ('magNormAGN', np.float),
                                   ('internalAVBulge', np.float),
                                   ('internalRVBulge', np.float),
                                   ('internalAVDisk', np.float),
                                   ('internalRVDisk', np.float),
                                   ('mag', ugrizy),
                                   ('magCalc', ugrizyE),
                                   ('Rbar', ugrizy),
                                   ('V', ugrizy),
                                   ('S_m02', ugrizy),
                                   ('S_p06', E),
                                   ('S_p10', E),
                                   ('photo_Rbar', ugrizy),   # `photo` indicates the photometrically
                                   ('photo_V', ugrizy),      # trained estimate.
                                   ('photo_S_m02', ugrizy),
                                   ('photo_S_p06', E),
                                   ('photo_S_p10', E),
                                   ('photo_redshift', np.float)])
    copy_fields = ['galTileID', 'objectID', 'raJ2000', 'decJ2000', 'redshift',
                   'sedPathBulge', 'sedPathDisk', 'sedPathAGN', 'magNormBulge',
                   'magNormDisk', 'magNormAGN', 'internalAVBulge', 'internalAVDisk',
                   'internalRVBulge', 'internalRVDisk', 'mag', 'magCalc',
                   'Rbar', 'V', 'S_m02', 'S_p06', 'S_p10']

    # copy input columns into output recarray
    for field in copy_fields:
        data[field] = test_objs[field]

    # just do weak-lensing relevant corrections...

    print 'training r-band centroid shifts'
    data['photo_Rbar']['LSST_r'] = ML(train_objs, test_objs,
                                      predict_var='Rbar', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(np.std(data['Rbar']['LSST_r'] - data['photo_Rbar']['LSST_r']))

    print 'training r-band zenith second-moment shifts'
    data['photo_V']['LSST_r'] = ML(train_objs, test_objs,
                                   predict_var='V', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(np.std(data['V']['LSST_r'] - data['photo_V']['LSST_r']))

    print 'training r-band seeing shifts'
    data['photo_S_m02']['LSST_r'] = ML(train_objs, test_objs,
                                       predict_var='S_m02', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(np.std(data['S_m02']['LSST_r'] - data['photo_S_m02']['LSST_r']))


    print 'training i-band centroid shifts'
    data['photo_Rbar']['LSST_i'] = ML(train_objs, test_objs,
                                      predict_var='Rbar', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(np.std(data['Rbar']['LSST_i'] - data['photo_Rbar']['LSST_i']))

    print 'training i-band zenith second-moment shifts'
    data['photo_V']['LSST_i'] = ML(train_objs, test_objs,
                                   predict_var='V', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(np.std(data['V']['LSST_i'] - data['photo_V']['LSST_i']))

    print 'training i-band seeing shifts'
    data['photo_S_m02']['LSST_i'] = ML(train_objs, test_objs,
                                       predict_var='S_m02', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(np.std(data['S_m02']['LSST_i'] - data['photo_S_m02']['LSST_i']))


    print 'training Euclid_350 diffraction limit shifts'
    data['photo_S_p06']['Euclid_350'] = ML(train_objs, test_objs,
                                           predict_var='S_p06',
                                           predict_band='Euclid_350', **kwargs)
    print 'resid std: {}'.format(np.std(data['S_p06']['Euclid_350']
                                           - data['photo_S_p06']['Euclid_350']))

    print 'training photometric redshifts'
    data['photo_redshift'] = ML(train_objs, test_objs,
                                predict_var='redshift', **kwargs)
    print 'resid std: {}'.format(np.std(data['redshift'] - data['photo_redshift']))
    print 'std((zphot - zspec)/(1+zspec)): {}'.format(
        np.std((data['photo_redshift'] - data['redshift'])/(1+data['redshift'])))

    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trainfile', default='output/galaxy_data.pkl',
                        help='file containing training data (Default: output/galaxy_data.pkl)')
    parser.add_argument('--testfile', default='output/galaxy_data.pkl',
                        help='file containing testing data (Default: output/galaxy_data.pkl)')
    parser.add_argument('--outfile', default='output/corrected_galaxy_data.pkl',
                        help='output file (Default: output/corrected_galaxy_data.pkl)')
    parser.add_argument('--trainstart', type=int, default=0,
                        help='object index at which to start training (Default: 0)')
    parser.add_argument('--ntrain', type=int, default=16000,
                        help='number of objects on which to train ML (Default: 16000)')
    parser.add_argument('--teststart', type=int, default=16000,
                        help='object index at which to start training (Default: 16000)')
    parser.add_argument('--ntest', type=int, default=8000,
                        help='number of objects on which to test ML (Default: 4000)')
    parser.add_argument('--use_color', action='store_true',
                        help="use only colors as features (Default: colors + 1 magnitude)")
    parser.add_argument('--use_mag', action='store_true',
                        help="use only magnitudes as features (Default: colors + 1 magnitude)")
    parser.add_argument('--no_err', action='store_true',
                        help="dont perturb magnitudes (Default: estimate LSST mag uncertainties)")
    regressor = parser.add_mutually_exclusive_group()
    regressor.add_argument("--RandomForest", action="store_true",
                           help="Use random forest regressor (Default: SVR)")
    regressor.add_argument("--ExtraTrees", action="store_true",
                           help="Use extra trees regressor (Default: SVR)")
    args = parser.parse_args()

    if args.RandomForest:
        regressor = 'RandomForest'
    elif args.ExtraTrees:
        regressor = 'ExtraTrees'
    else:
        regressor = 'SVR'

    print "loading data"
    train_objs = cPickle.load(open(args.trainfile))
    test_objs = cPickle.load(open(args.testfile))

    if not args.no_err:
        shape = len(test_objs['magCalc']['LSST_u'])
        for i, band in enumerate('ugrizy'):
            test_objs['magCalc']['LSST_'+band] += (np.random.randn(shape)
                                                   * test_objs['magErr']['LSST_'+band])

    out = gal_ML(train_objs[args.trainstart:args.trainstart+args.ntrain],
                 test_objs[args.teststart:args.teststart+args.ntest],
                 use_color=args.use_color,
                 use_mag=args.use_mag,
                 regressor=regressor)
    cPickle.dump(out, open(args.outfile, 'wb'))
