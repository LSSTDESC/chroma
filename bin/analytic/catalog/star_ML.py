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

import regressor

def ML(train_objs, test_objs, predict_var=None, predict_band=None,
        use_color=False, use_mag=False):

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

    import sklearn.svm
    learner = regressor.regress(sklearn.svm.SVR(C=100, gamma=0.1))
    # import sklearn.ensemble
    # learner = regressor.regress(sklearn.ensemble.RandomForestRegressor(50))
    learner.add_training_data(train_X, train_Y)
    learner.train()
    predict_Y = learner.predict(test_X)

    return predict_Y

def star_ML(train_objs, test_objs, **kwargs):
    ugrizy = [('LSST_u', np.float32),
              ('LSST_g', np.float32),
              ('LSST_r', np.float32),
              ('LSST_i', np.float32),
              ('LSST_z', np.float32),
              ('LSST_y', np.float32)]
    ugrizyE = [('LSST_u', np.float32),
               ('LSST_g', np.float32),
               ('LSST_r', np.float32),
               ('LSST_i', np.float32),
               ('LSST_z', np.float32),
               ('LSST_y', np.float32),
               ('Euclid_150', np.float32),
               ('Euclid_250', np.float32),
               ('Euclid_350', np.float32),
               ('Euclid_450', np.float32)]
    E = [('Euclid_150', np.float32),
         ('Euclid_250', np.float32),
         ('Euclid_350', np.float32),
         ('Euclid_450', np.float32)]

    data = np.recarray((len(test_objs),),
                          dtype = [('objectID', np.uint64),
                                   ('raJ2000', np.float64),
                                   ('decJ2000', np.float64),
                                   ('magNorm', np.float32),
                                   ('sedFilePath', np.str_, 64),
                                   ('galacticAv', np.float32),
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
                                   ('photo_S_p10', E)])
    copy_fields = ['objectID', 'raJ2000', 'decJ2000', 'magNorm',
                   'sedFilePath', 'galacticAv', 'mag', 'magCalc',
                   'Rbar', 'V', 'S_m02', 'S_p06', 'S_p10']

    # copy input columns into output recarray
    for field in copy_fields:
        data[field] = test_objs[field]

    # just do WL relevant corrections...

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
                                           predict_var='S_p06', predict_band='Euclid_350', **kwargs)
    print 'resid std: {}'.format(np.std(data['S_p06']['Euclid_350']
                                        - data['photo_S_p06']['Euclid_350']))

    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trainfile', default='output/star_data.pkl',
                        help='file containing training data (Default: output/star_data.pkl)')
    parser.add_argument('--testfile', default='output/star_data.pkl',
                        help='file containing testing data (Default: output/star_data.pkl)')
    parser.add_argument('--outfile', default='output/corrected_star_data.pkl',
                        help="output file (Default: output/corrected_star_data.pkl)")
    parser.add_argument('--trainstart', type=int, default=0,
                        help='object index at which to start training (Default: 0)')
    parser.add_argument('--ntrain', type=int, default=16000,
                        help='number of objects on which to train ML (Default: 16000)')
    parser.add_argument('--teststart', type=int, default=16000,
                        help='object index at which to start training (Default: 16000)')
    parser.add_argument('--ntest', type=int, default=4000,
                        help='number of objects on which to test ML (Default: 4000)')
    parser.add_argument('--use_color', action='store_true',
                        help="use only colors as features (Default: colors + 1 magnitude)")
    parser.add_argument('--use_mag', action='store_true',
                        help="use only magnitudes as features (Default: colors + 1 magnitude)")
    parser.add_argument('--no_err', action='store_true',
                        help="dont perturb magnitudes (Default: estimate LSST mag uncertainties)")
    args = parser.parse_args()

    train_objs = cPickle.load(open(args.trainfile))
    test_objs = cPickle.load(open(args.testfile))

    if not args.no_err:
        shape = len(test_objs['magCalc']['LSST_u'])
        for i, band in enumerate('ugrizy'):
            test_objs['magCalc']['LSST_'+band] += (np.random.randn(shape)
                                                   * test_objs['magErr']['LSST_'+band])

    out = star_ML(train_objs[args.trainstart:args.trainstart+args.ntrain],
                  test_objs[args.teststart:args.teststart+args.ntest],
                  use_color=args.use_color,
                  use_mag=args.use_mag)
    cPickle.dump(out, open(args.outfile, 'wb'))
