import pickle

import numpy

def coallesce():
    starlib = numpy.zeros(16, dtype=[('type', 'a10'),
                                     ('filter', 'a10'),
                                     ('FWHM', 'f4', 8),
                                     ('beta', 'f4', 8),
                                     ('q', 'f4', 8),
                                     ('phi', 'f4', 8),
                                     ('flux', 'f4', 8)])
    # fill in starlib here...
    a = pickle.load(open('pickles/star.r.pik'))
    for i in range(8):
        starlib[i]['type'] = a[8*i]['type']
        starlib[i]['filter'] = 'r'
        for j in range(8):
            starlib[i]['FWHM'][j] = a[8*i+j]['fwhm']
            starlib[i]['beta'][j] = a[8*i+j]['beta']
            starlib[i]['q'][j] = a[8*i+j]['q']
            starlib[i]['phi'][j] = a[8*i+j]['phi']
            starlib[i]['flux'][j] = a[8*i+j]['flux']
    # fill in starlib here...
    a = pickle.load(open('pickles/star.i.pik'))
    for i in range(8):
        starlib[8+i]['type'] = a[8*i]['type']
        starlib[8+i]['filter'] = 'i'
        for j in range(8):
            starlib[8+i]['FWHM'][j] = a[8*i+j]['fwhm']
            starlib[8+i]['beta'][j] = a[8*i+j]['beta']
            starlib[8+i]['q'][j] = a[8*i+j]['q']
            starlib[8+i]['phi'][j] = a[8*i+j]['phi']
            starlib[8+i]['flux'][j] = a[8*i+j]['flux']

    G5vlib = numpy.zeros(16, dtype=[('type', 'a10'),
                                    ('filter', 'a10'),
                                    ('FWHM', 'f4', 8),
                                    ('beta', 'f4', 8),
                                    ('q', 'f4', 8),
                                    ('phi', 'f4', 8),
                                    ('flux', 'f4', 8)])
    # fill in G5vlib here...
    a = pickle.load(open('pickles/G5v.r.pik'))
    for i in range(8):
        G5vlib[i]['type'] = a[8*i]['type']
        G5vlib[i]['filter'] = 'r'
        for j in range(8):
            G5vlib[i]['FWHM'][j] = a[8*i+j]['fwhm']
            G5vlib[i]['beta'][j] = a[8*i+j]['beta']
            G5vlib[i]['q'][j] = a[8*i+j]['q']
            G5vlib[i]['phi'][j] = a[8*i+j]['phi']
            G5vlib[i]['flux'][j] = a[8*i+j]['flux']
    # fill in G5vlib here...
    a = pickle.load(open('pickles/G5v.i.pik'))
    for i in range(8):
        G5vlib[8+i]['type'] = a[8*i]['type']
        G5vlib[8+i]['filter'] = 'i'
        for j in range(8):
            G5vlib[8+i]['FWHM'][j] = a[8*i+j]['fwhm']
            G5vlib[8+i]['beta'][j] = a[8*i+j]['beta']
            G5vlib[8+i]['q'][j] = a[8*i+j]['q']
            G5vlib[8+i]['phi'][j] = a[8*i+j]['phi']
            G5vlib[8+i]['flux'][j] = a[8*i+j]['flux']

    gallib = numpy.zeros(1600, dtype=[('type', 'a10'),
                                      ('filter', 'a10'),
                                      ('z','f4'),
                                      ('FWHM', 'f4', 8),
                                      ('beta', 'f4', 8),
                                      ('q', 'f4', 8),
                                      ('phi', 'f4', 8),
                                      ('flux', 'f4', 8)])

    k = 0
    for z in numpy.arange(0.0, 3.0, 0.03):
        # fill in G5vlib here...
        a = pickle.load(open('pickles/gal.r.{:02d}.pik'.format(int(round(z / 0.03)))))
        if len(a) == 64:
            for i in range(8):
                gallib[k]['type'] = a[8*i]['type']
                gallib[k]['filter'] = 'r'
                gallib[k]['z'] = z
                for j in range(8):
                    gallib[k]['FWHM'][j] = a[8*i+j]['fwhm']
                    gallib[k]['beta'][j] = a[8*i+j]['beta']
                    gallib[k]['q'][j] = a[8*i+j]['q']
                    gallib[k]['phi'][j] = a[8*i+j]['phi']
                    gallib[k]['flux'][j] = a[8*i+j]['flux']
                k += 1
        # fill in gallib here...
        a = pickle.load(open('pickles/gal.i.{:02d}.pik'.format(int(round(z / 0.03)))))
        if len(a) == 64:
            for i in range(8):
                gallib[k]['type'] = a[8*i]['type']
                gallib[k]['filter'] = 'i'
                gallib[k]['z'] = z
                for j in range(8):
                    gallib[k]['FWHM'][j] = a[8*i+j]['fwhm']
                    gallib[k]['beta'][j] = a[8*i+j]['beta']
                    gallib[k]['q'][j] = a[8*i+j]['q']
                    gallib[k]['phi'][j] = a[8*i+j]['phi']
                    gallib[k]['flux'][j] = a[8*i+j]['flux']
                k += 1
    pickle.dump((G5vlib, starlib, gallib), open('data.pik', 'w'))

if __name__ == '__main__':
    coallesce()
