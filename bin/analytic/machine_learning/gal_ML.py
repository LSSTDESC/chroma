import cPickle
from argparse import ArgumentParser

import numpy
import regressor

def ML(train_objs, test_objs, predict_var=None, predict_band=None,
        use_color=False, use_colormag=False):

    if predict_var is None:
        raise ValueError
    ntrain = len(train_objs)
    ntest = len(test_objs)
    train_Y = numpy.empty(ntrain, dtype=numpy.float)
    test_Y = numpy.empty(ntest, dtype=numpy.float)
    if predict_band is None:
        train_Y[:] = train_objs[predict_var]
        test_Y[:] = test_objs[predict_var]
    else:
        train_Y[:] = train_objs[predict_var][predict_band]
        test_Y[:] = test_objs[predict_var][predict_band]

    if use_colormag: # use i-band magnitude, and five colors to train
        # r-band, V
        train_X = numpy.empty([ntrain, 6], dtype=numpy.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u'] - train_objs['magCalc']['LSST_g']
        train_X[:,1] = train_objs['magCalc']['LSST_g'] - train_objs['magCalc']['LSST_r']
        train_X[:,2] = train_objs['magCalc']['LSST_r'] - train_objs['magCalc']['LSST_i']
        train_X[:,3] = train_objs['magCalc']['LSST_i'] - train_objs['magCalc']['LSST_z']
        train_X[:,4] = train_objs['magCalc']['LSST_z'] - train_objs['magCalc']['LSST_y']
        train_X[:,5] = train_objs['magCalc']['LSST_i']

        # test data
        test_X = numpy.empty([ntest, 6], dtype=numpy.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u'] - test_objs['magCalc']['LSST_g']
        test_X[:,1] = test_objs['magCalc']['LSST_g'] - test_objs['magCalc']['LSST_r']
        test_X[:,2] = test_objs['magCalc']['LSST_r'] - test_objs['magCalc']['LSST_i']
        test_X[:,3] = test_objs['magCalc']['LSST_i'] - test_objs['magCalc']['LSST_z']
        test_X[:,4] = test_objs['magCalc']['LSST_z'] - test_objs['magCalc']['LSST_y']
        test_X[:,5] = test_objs['magCalc']['LSST_i']
    elif use_color: # only use five colors to train
        # r-band, V
        train_X = numpy.empty([ntrain, 5], dtype=numpy.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u'] - train_objs['magCalc']['LSST_g']
        train_X[:,1] = train_objs['magCalc']['LSST_g'] - train_objs['magCalc']['LSST_r']
        train_X[:,2] = train_objs['magCalc']['LSST_r'] - train_objs['magCalc']['LSST_i']
        train_X[:,3] = train_objs['magCalc']['LSST_i'] - train_objs['magCalc']['LSST_z']
        train_X[:,4] = train_objs['magCalc']['LSST_z'] - train_objs['magCalc']['LSST_y']

        # test data
        test_X = numpy.empty([ntest, 5], dtype=numpy.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u'] - test_objs['magCalc']['LSST_g']
        test_X[:,1] = test_objs['magCalc']['LSST_g'] - test_objs['magCalc']['LSST_r']
        test_X[:,2] = test_objs['magCalc']['LSST_r'] - test_objs['magCalc']['LSST_i']
        test_X[:,3] = test_objs['magCalc']['LSST_i'] - test_objs['magCalc']['LSST_z']
        test_X[:,4] = test_objs['magCalc']['LSST_z'] - test_objs['magCalc']['LSST_y']
    else: #default use just magnitudes to train
        # r-band, V
        train_X = numpy.empty([ntrain, 6], dtype=numpy.float)

        # training data
        train_X[:,0] = train_objs['magCalc']['LSST_u']
        train_X[:,1] = train_objs['magCalc']['LSST_g']
        train_X[:,2] = train_objs['magCalc']['LSST_r']
        train_X[:,3] = train_objs['magCalc']['LSST_i']
        train_X[:,4] = train_objs['magCalc']['LSST_z']
        train_X[:,5] = train_objs['magCalc']['LSST_y']

        # test data
        test_X = numpy.empty([ntest, 6], dtype=numpy.float)
        test_X[:,0] = test_objs['magCalc']['LSST_u']
        test_X[:,1] = test_objs['magCalc']['LSST_g']
        test_X[:,2] = test_objs['magCalc']['LSST_r']
        test_X[:,3] = test_objs['magCalc']['LSST_i']
        test_X[:,4] = test_objs['magCalc']['LSST_z']
        test_X[:,5] = test_objs['magCalc']['LSST_y']

    # import sklearn.svm
    # learner = regressor.regress(sklearn.svm.SVR())
    import sklearn.ensemble
    learner = regressor.regress(sklearn.ensemble.RandomForestRegressor(50))
    learner.add_training_data(train_X, train_Y)
    learner.train()
    predict_Y = learner.predict(test_X)

    return predict_Y

def ML_all(train_objs, test_objs, **kwargs):

    # do some pruning of troublesome objects
    good = numpy.ones(train_objs.shape, dtype=numpy.bool)
    for f in 'ugrizy':
        good = good & numpy.isfinite(train_objs['magCalc']['LSST_{}'.format(f)])
        good = good & numpy.isfinite(train_objs['magCalc']['LSST_{}'.format(f)])
    #     if predict_band is not None:
    #         good = good & numpy.isfinite(train_objs[predict_var]['LSST_{}'.format(f)])
    # if predict_band is None:
    #     good = good & numpy.isfinite(train_objs[predict_var])
    # kill everything with an AGN for now since mags don't match well.
    good = good & numpy.isnan(train_objs.magNormAGN)
    train_objs = train_objs[good]

    good = numpy.ones(test_objs.shape, dtype=numpy.bool)
    for f in 'ugrizy':
        good = good & numpy.isfinite(test_objs['magCalc']['LSST_{}'.format(f)])
        good = good & numpy.isfinite(test_objs['magCalc']['LSST_{}'.format(f)])
    #     if predict_band is not None:
    #         good = good & numpy.isfinite(test_objs[predict_var]['LSST_{}'.format(f)])
    # if predict_band is None:
    #     good = good & numpy.isfinite(test_objs[predict_var])
    # kill everything with an AGN for now since mags don't match well.
    good = good & numpy.isnan(test_objs.magNormAGN)
    test_objs = test_objs[good]

    ugrizy = [('LSST_u', numpy.float32),
              ('LSST_g', numpy.float32),
              ('LSST_r', numpy.float32),
              ('LSST_i', numpy.float32),
              ('LSST_z', numpy.float32),
              ('LSST_y', numpy.float32)]
    ugrizyE = [('LSST_u', numpy.float32),
               ('LSST_g', numpy.float32),
               ('LSST_r', numpy.float32),
               ('LSST_i', numpy.float32),
               ('LSST_z', numpy.float32),
               ('LSST_y', numpy.float32),
               ('Euclid_150', numpy.float32),
               ('Euclid_250', numpy.float32),
               ('Euclid_350', numpy.float32),
               ('Euclid_450', numpy.float32)]
    E = [('Euclid_150', numpy.float32),
         ('Euclid_250', numpy.float32),
         ('Euclid_350', numpy.float32),
         ('Euclid_450', numpy.float32)]

    data = numpy.recarray((len(test_objs),),
                          dtype = [('galTileID', numpy.uint64),
                                   ('objectID', numpy.uint64),
                                   ('raJ2000', numpy.float64),
                                   ('decJ2000', numpy.float64),
                                   ('redshift', numpy.float32),
                                   ('sedPathBulge', numpy.str_, 64),
                                   ('sedPathDisk', numpy.str_, 64),
                                   ('sedPathAGN', numpy.str_, 64),
                                   ('magNormBulge', numpy.float32),
                                   ('magNormDisk', numpy.float32),
                                   ('magNormAGN', numpy.float32),
                                   ('internalAVBulge', numpy.float32),
                                   ('internalRVBulge', numpy.float32),
                                   ('internalAVDisk', numpy.float32),
                                   ('internalRVDisk', numpy.float32),
                                   ('mag', ugrizy),
                                   ('magCalc', ugrizyE),
                                   ('R', ugrizy),
                                   ('V', ugrizy),
                                   ('S_m02', ugrizy),
                                   ('S_p06', E),
                                   ('S_p10', E),
                                   ('photo_R', ugrizy),
                                   ('photo_V', ugrizy),
                                   ('photo_S_m02', ugrizy),
                                   ('photo_S_p06', E),
                                   ('photo_S_p10', E),
                                   ('photo_redshift', numpy.float32)])
    copy_fields = ['galTileID', 'objectID', 'raJ2000', 'decJ2000', 'redshift',
                   'sedPathBulge', 'sedPathDisk', 'sedPathAGN', 'magNormBulge',
                   'magNormDisk', 'magNormAGN', 'internalAVBulge', 'internalAVDisk',
                   'internalRVBulge', 'internalRVDisk', 'mag', 'magCalc',
                   'R', 'V', 'S_m02', 'S_p06', 'S_p10']

    # poor man's rec_array extension
    for field in copy_fields:
        data[field] = test_objs[field]

    # so don't actually do all, just do WL relevant corrections...

    print 'training r-band centroid shifts'
    data['photo_R']['LSST_r'] = ML(train_objs, test_objs,
                                   predict_var='R', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['R']['LSST_r'] - data['photo_R']['LSST_r']))

    print 'training r-band zenith second-moment shifts'
    data['photo_V']['LSST_r'] = ML(train_objs, test_objs,
                                   predict_var='V', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['V']['LSST_r'] - data['photo_V']['LSST_r']))

    print 'training r-band seeing shifts'
    data['photo_S_m02']['LSST_r'] = ML(train_objs, test_objs,
                                       predict_var='S_m02', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_m02']['LSST_r'] - data['photo_S_m02']['LSST_r']))


    print 'training i-band centroid shifts'
    data['photo_R']['LSST_i'] = ML(train_objs, test_objs,
                                   predict_var='R', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['R']['LSST_i'] - data['photo_R']['LSST_i']))

    print 'training i-band zenith second-moment shifts'
    data['photo_V']['LSST_i'] = ML(train_objs, test_objs,
                                   predict_var='V', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['V']['LSST_i'] - data['photo_V']['LSST_i']))

    print 'training i-band seeing shifts'
    data['photo_S_m02']['LSST_i'] = ML(train_objs, test_objs,
                                       predict_var='S_m02', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_m02']['LSST_i'] - data['photo_S_m02']['LSST_i']))


    print 'training Euclid_350 diffraction limit shifts'
    data['photo_S_p06']['Euclid_350'] = ML(train_objs, test_objs,
                                           predict_var='S_p06',
                                           predict_band='Euclid_350', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_p06']['Euclid_350']
                                           - data['photo_S_p06']['Euclid_350']))

    print 'training photometric redshifts'
    data['photo_redshift'] = ML(train_objs, test_objs,
                                predict_var='redshift', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['redshift'] - data['photo_redshift']))
    print 'std((zphot - zspec)/(1+zspec)): {}'.format(
        numpy.std((data['photo_redshift'] - data['redshift'])/(1+data['redshift'])))

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
    parser.add_argument('--ntest', type=int, default=4000,
                        help='number of objects on which to test ML (Default: 4000)')
    parser.add_argument('--use_color', action='store_true',
                        help="use colors as features (Default: magnitudes)")
    parser.add_argument('--use_colormag', action='store_true',
                        help="use colors + one magnitude as features (Default: magnitudes)")
    args = parser.parse_args()

    train_objs = cPickle.load(open(args.trainfile))
    test_objs = cPickle.load(open(args.testfile))

    out = ML_all(train_objs[args.trainstart:args.trainstart+args.ntrain],
                 test_objs[args.teststart:args.teststart+args.ntest],
                 use_color=args.use_color,
                 use_colormag=args.use_colormag)
    cPickle.dump(out, open(args.outfile, 'wb'))
