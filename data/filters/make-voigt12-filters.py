import numpy as np

def voigt_filters():
    central_wavelength = 725
    widths = np.array([150.0, 250.0, 350.0, 450.0])
    waves = np.arange(450.0, 1000.0, 0.1)
    for width in widths:
        fil = open("voigt12_{:03d}.dat".format(int(width)), 'w')
        for w in waves:
            if w > central_wavelength - width/2 and w < central_wavelength + width/2:
                fil.write("{:5.1f}  {:f}\n".format(w, 1.0))
            else:
                fil.write("{:5.1f}  {:f}\n".format(w, 0.0))
        fil.close()

voigt_filters()
