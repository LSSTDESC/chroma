import numpy

def mono_spec(wave):
    waves = numpy.arange(300.0, 1100.1, 0.1)
    flux = numpy.zeros_like(waves)
    flux[numpy.abs(waves - 500.0) < 0.001] = 1.0
    flux[numpy.abs(waves - wave) < 0.001] = 1.e6
    return waves, flux

mono_waves = numpy.arange(300.0, 1100.1, 25)
for wave in mono_waves:
    waves, flux = mono_spec(wave)
    with open('mono.{:d}.spec'.format(int(round(wave))), 'w') as fil:
        for w, f in zip(waves, flux):
            fil.write('{} {}\n'.format(w, f))
