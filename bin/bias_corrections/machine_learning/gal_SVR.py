import cPickle
from argparse import ArgumentParser

import numpy
import SVR_correction

def SVR(objs, ntrain, ntest, predict_var=None, predict_band=None, use_color=False, use_colormag=False):
    if predict_var is None:
        raise ValueError
    train_Y = numpy.empty([ntrain], dtype=numpy.float)
    test_Y = numpy.empty([ntest], dtype=numpy.float)
    if predict_band is None:
        train_Y[:] = objs[predict_var][0:ntrain]
        test_Y[:] = objs[predict_var][ntrain:ntrain+ntest]
    else:
        train_Y[:] = objs[predict_var][predict_band][0:ntrain]
        test_Y[:] = objs[predict_var][predict_band][ntrain:ntrain+ntest]

    if use_colormag:
        # r-band, V
        train_X = numpy.empty([ntrain, 6], dtype=numpy.float)

        #training data
        train_X[:,0] = objs['magCalc']['LSST_u'][0:ntrain] - objs['magCalc']['LSST_g'][0:ntrain]
        train_X[:,1] = objs['magCalc']['LSST_g'][0:ntrain] - objs['magCalc']['LSST_r'][0:ntrain]
        train_X[:,2] = objs['magCalc']['LSST_r'][0:ntrain] - objs['magCalc']['LSST_i'][0:ntrain]
        train_X[:,3] = objs['magCalc']['LSST_i'][0:ntrain] - objs['magCalc']['LSST_z'][0:ntrain]
        train_X[:,4] = objs['magCalc']['LSST_z'][0:ntrain] - objs['magCalc']['LSST_y'][0:ntrain]
        train_X[:,5] = objs['magCalc']['LSST_i'][0:ntrain]

        #test data
        test_X = numpy.empty([ntest, 6], dtype=numpy.float)
        test_X[:,0] = objs['magCalc']['LSST_u'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_g'][ntrain:ntrain+ntest]
        test_X[:,1] = objs['magCalc']['LSST_g'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_r'][ntrain:ntrain+ntest]
        test_X[:,2] = objs['magCalc']['LSST_r'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_i'][ntrain:ntrain+ntest]
        test_X[:,3] = objs['magCalc']['LSST_i'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_z'][ntrain:ntrain+ntest]
        test_X[:,4] = objs['magCalc']['LSST_z'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_y'][ntrain:ntrain+ntest]
        test_X[:,5] = objs['magCalc']['LSST_i'][ntrain:ntrain+ntest]
    elif use_color:
        # r-band, V
        train_X = numpy.empty([ntrain, 5], dtype=numpy.float)

        #training data
        train_X[:,0] = objs['magCalc']['LSST_u'][0:ntrain] - objs['magCalc']['LSST_g'][0:ntrain]
        train_X[:,1] = objs['magCalc']['LSST_g'][0:ntrain] - objs['magCalc']['LSST_r'][0:ntrain]
        train_X[:,2] = objs['magCalc']['LSST_r'][0:ntrain] - objs['magCalc']['LSST_i'][0:ntrain]
        train_X[:,3] = objs['magCalc']['LSST_i'][0:ntrain] - objs['magCalc']['LSST_z'][0:ntrain]
        train_X[:,4] = objs['magCalc']['LSST_z'][0:ntrain] - objs['magCalc']['LSST_y'][0:ntrain]

        #test data
        test_X = numpy.empty([ntest, 5], dtype=numpy.float)
        test_X[:,0] = objs['magCalc']['LSST_u'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_g'][ntrain:ntrain+ntest]
        test_X[:,1] = objs['magCalc']['LSST_g'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_r'][ntrain:ntrain+ntest]
        test_X[:,2] = objs['magCalc']['LSST_r'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_i'][ntrain:ntrain+ntest]
        test_X[:,3] = objs['magCalc']['LSST_i'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_z'][ntrain:ntrain+ntest]
        test_X[:,4] = objs['magCalc']['LSST_z'][ntrain:ntrain+ntest] - objs['magCalc']['LSST_y'][ntrain:ntrain+ntest]
    else:
        # r-band, V
        train_X = numpy.empty([ntrain, 6], dtype=numpy.float)

        #training data
        train_X[:,0] = objs['magCalc']['LSST_u'][0:ntrain]
        train_X[:,1] = objs['magCalc']['LSST_g'][0:ntrain]
        train_X[:,2] = objs['magCalc']['LSST_r'][0:ntrain]
        train_X[:,3] = objs['magCalc']['LSST_i'][0:ntrain]
        train_X[:,4] = objs['magCalc']['LSST_z'][0:ntrain]
        train_X[:,5] = objs['magCalc']['LSST_y'][0:ntrain]

        #test data
        test_X = numpy.empty([ntest, 6], dtype=numpy.float)
        test_X[:,0] = objs['magCalc']['LSST_u'][ntrain:ntrain+ntest]
        test_X[:,1] = objs['magCalc']['LSST_g'][ntrain:ntrain+ntest]
        test_X[:,2] = objs['magCalc']['LSST_r'][ntrain:ntrain+ntest]
        test_X[:,3] = objs['magCalc']['LSST_i'][ntrain:ntrain+ntest]
        test_X[:,4] = objs['magCalc']['LSST_z'][ntrain:ntrain+ntest]
        test_X[:,5] = objs['magCalc']['LSST_y'][ntrain:ntrain+ntest]

    learner = SVR_correction.SV_regress()
    learner.add_training_data(train_X, train_Y)
    learner.train()
    predict_Y = learner.predict(test_X)

    return predict_Y

def SVR_all(objs, ntrain, ntest, **kwargs):
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

    data = numpy.recarray((ntest,),
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
        data[field] = objs[field][ntrain:ntrain+ntest]

    # so don't actually do all, just do WL relevant corrections...

    print 'training r-band centroid shifts'
    data['photo_R']['LSST_r'] = SVR(objs, ntrain, ntest, predict_var='R', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['R']['LSST_r'] - data['photo_R']['LSST_r']))

    print 'training r-band zenith second-moment shifts'
    data['photo_V']['LSST_r'] = SVR(objs, ntrain, ntest, predict_var='V', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['V']['LSST_r'] - data['photo_V']['LSST_r']))

    print 'training r-band seeing shifts'
    data['photo_S_m02']['LSST_r'] = SVR(objs, ntrain, ntest, predict_var='S_m02', predict_band='LSST_r', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_m02']['LSST_r'] - data['photo_S_m02']['LSST_r']))


    print 'training i-band centroid shifts'
    data['photo_R']['LSST_i'] = SVR(objs, ntrain, ntest, predict_var='R', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['R']['LSST_i'] - data['photo_R']['LSST_i']))

    print 'training i-band zenith second-moment shifts'
    data['photo_V']['LSST_i'] = SVR(objs, ntrain, ntest, predict_var='V', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['V']['LSST_i'] - data['photo_V']['LSST_i']))

    print 'training i-band seeing shifts'
    data['photo_S_m02']['LSST_i'] = SVR(objs, ntrain, ntest, predict_var='S_m02', predict_band='LSST_i', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_m02']['LSST_i'] - data['photo_S_m02']['LSST_i']))


    print 'training Euclid_350 diffraction limit shifts'
    data['photo_S_p06']['Euclid_350'] = SVR(objs, ntrain, ntest, predict_var='S_p06',
                                            predict_band='Euclid_350', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['S_p06']['Euclid_350'] - data['photo_S_p06']['Euclid_350']))

    print 'training photometric redshifts'
    data['photo_redshift'] = SVR(objs, ntrain, ntest, predict_var='redshift', **kwargs)
    print 'resid std: {}'.format(numpy.std(data['redshift'] - data['photo_redshift']))
    print 'std((zphot - zspec)/(1+zspec)): {}'.format(
        numpy.std((data['photo_redshift'] - data['redshift'])/(1+data['redshift'])))

    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--infile', default='galaxy_data.pkl',
                        help="input file")
    parser.add_argument('--outfile', default='corrected_galaxy_data.pkl',
                        help="output file")
    parser.add_argument('--ntrain', type=int, default=16000,
                        help="number of objects on which to train SVR")
    parser.add_argument('--ntest', type=int, default=4000,
                        help="number of objects on which to test SVR")
    parser.add_argument('--use_color', action='store_true',
                        help="use colors instead of magnitudes as predictors")
    parser.add_argument('--use_colormag', action='store_true',
                        help="use colors and one magnitude as predictors")
    args = parser.parse_args()

    gals = cPickle.load(open(args.infile))

    # kill everything with an AGN for now since mags don't match well.
    w = numpy.isnan(gals.magNormAGN)
    gals = gals[w]
    out = SVR_all(gals, ntrain=args.ntrain, ntest=args.ntest, use_color=args.use_color, use_colormag=args.use_colormag)
    cPickle.dump(out, open(args.outfile, 'wb'))
