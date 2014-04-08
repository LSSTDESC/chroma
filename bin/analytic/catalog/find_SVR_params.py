import cPickle
from argparse import ArgumentParser

import numpy as np

import regressor

def ML(train_objs, test_objs, predict_var=None, predict_band=None,
        use_color=False, use_mag=False, C=None, gamma=None, epsilon=None):
    if C is None:
        C = 100
    if gamma is None:
        gamma = 0.1
    if epsilon is None:
        epsilon = 0.1
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
    learner = regressor.regress(sklearn.svm.SVR(C=C, gamma=gamma))
    # import sklearn.ensemble
    # learner = regressor.regress(sklearn.ensemble.RandomForestRegressor(50))
    learner.add_training_data(train_X, train_Y)
    learner.train()
    predict_Y = learner.predict(test_X)

    return predict_Y

def gal_ML(train_objs, test_objs, predict_var=None, predict_band=None, **kwargs):

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
                          dtype = [('galTileID', np.uint64),
                                   ('objectID', np.uint64),
                                   ('raJ2000', np.float64),
                                   ('decJ2000', np.float64),
                                   ('redshift', np.float32),
                                   ('sedPathBulge', np.str_, 64),
                                   ('sedPathDisk', np.str_, 64),
                                   ('sedPathAGN', np.str_, 64),
                                   ('magNormBulge', np.float32),
                                   ('magNormDisk', np.float32),
                                   ('magNormAGN', np.float32),
                                   ('internalAVBulge', np.float32),
                                   ('internalRVBulge', np.float32),
                                   ('internalAVDisk', np.float32),
                                   ('internalRVDisk', np.float32),
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
                                   ('photo_redshift', np.float32)])
    copy_fields = ['galTileID', 'objectID', 'raJ2000', 'decJ2000', 'redshift',
                   'sedPathBulge', 'sedPathDisk', 'sedPathAGN', 'magNormBulge',
                   'magNormDisk', 'magNormAGN', 'internalAVBulge', 'internalAVDisk',
                   'internalRVBulge', 'internalRVDisk', 'mag', 'magCalc',
                   'Rbar', 'V', 'S_m02', 'S_p06', 'S_p10']

    # copy input columns into output recarray
    for field in copy_fields:
        data[field] = test_objs[field]

    # investigate a range of C and gamma values
    C_range = 10.0**np.arange(-2, 4)
    gamma_range = 10.0**np.arange(-4, 3)
    out = np.zeros((len(C_range), len(gamma_range)), dtype=float)
    for i, C in enumerate(C_range):
        for j, gamma in enumerate(gamma_range):
            predict = ML(train_objs, test_objs, predict_var=predict_var, predict_band=predict_band,
                         C=C, gamma=gamma, **kwargs)
            out[i,j] = np.std(predict - data[predict_var][predict_band])
            print 'C: {}, gamma: {}, std: {}'.format(C, gamma, out[i,j])
    import pylab as pl
    pl.imshow(out)
    pl.xticks(np.arange(len(gamma_range), gamma_range, rotation=45)
    pl.yticks(np.arange(len(C_range), C_range)
    pl.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--trainfile', default='output/galaxy_data.pkl',
                        help='file containing training data (Default: output/galaxy_data.pkl)')
    parser.add_argument('--testfile', default='output/galaxy_data.pkl',
                        help='file containing testing data (Default: output/galaxy_data.pkl)')
    parser.add_argument('--trainstart', type=int, default=0,
                        help='object index at which to start training (Default: 0)')
    parser.add_argument('--ntrain', type=int, default=1000,
                        help='number of objects on which to train ML (Default: 1000)')
    parser.add_argument('--teststart', type=int, default=1000,
                        help='object index at which to start training (Default: 1000)')
    parser.add_argument('--ntest', type=int, default=4000,
                        help='number of objects on which to test ML (Default: 4000)')
    parser.add_argument('--use_color', action='store_true',
                        help="use only colors as features (Default: colors + 1 magnitude)")
    parser.add_argument('--use_mag', action='store_true',
                        help="use only magnitudes as features (Default: colors + 1 magnitude)")
    parser.add_argument('--predict_var', type=str, default='Rbar')
    parser.add_argument('--predict_band', type=str, default='LSST_r')
    args = parser.parse_args()

    train_objs = cPickle.load(open(args.trainfile))
    test_objs = cPickle.load(open(args.testfile))

    out = gal_ML(train_objs[args.trainstart:args.trainstart+args.ntrain],
                 test_objs[args.teststart:args.teststart+args.ntest],
                 use_color=args.use_color,
                 use_mag=args.use_mag,
                 predict_var=args.predict_var,
                 predict_band=args.predict_band)
