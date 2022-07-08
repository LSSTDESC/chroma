import _mypath
import numpy as np

def test_Sersic():
    from chroma.Sersic import Sersic
    '''Creating a test case for the various ways of initializing Sersics'''
    y0 = 0.1
    x0 = 0.3
    n = 1.5
    peak = 1.0

    # a, b, phi
    phi = 0.1
    a = 0.5
    b = 0.2
    s1 = Sersic(y0, x0, n, peak=peak, a=a, b=b, phi=phi)
    print('It is an error if following 9 or so values do not all match')
    print(s1(0.2, 2.1))

    # C11, C12, C22
    C11 = (np.cos(phi)**2.0 / a**2.0 + np.sin(phi)**2.0 / b**2.0)
    C22 = (np.sin(phi)**2.0 / a**2.0 + np.cos(phi)**2.0 / b**2.0)
    C12 = 0.5 * (1.0/a**2.0 - 1.0/b**2.0) * np.sin(2.0 * phi)
    s2 = Sersic(y0, x0, n, peak=peak, C11=C11, C12=C12, C22=C22)
    print(s2(0.2, 2.1))

    # r_e, b_over_a, phi
    r_e = np.sqrt(a * b)
    b_over_a = b / a
    s3 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print(s3(0.2, 2.1))

    # r_e, emag, phi
    emag = (a**2.0 - b**2.0) / (a**2.0 + b**2.0)
    s4 = Sersic(y0, x0, n, peak=peak, r_e=r_e, emag=emag, phi=phi)
    print(s4(0.2, 2.1))

    # r_e, gmag, phi
    gmag = (a - b) / (a + b)
    s5 = Sersic(y0, x0, n, peak=peak, r_e=r_e, gmag=gmag, phi=phi)
    print(s5(0.2, 2.1))

    # r_e, e1, e2
    e1 = emag * np.cos(2.0 * phi)
    e2 = emag * np.sin(2.0 * phi)
    s6 = Sersic(y0, x0, n, peak=peak, r_e=r_e, e1=e1, e2=e2)
    print(s6(0.2, 2.1))

    # r_e, g1, g2
    g1 = gmag * np.cos(2.0 * phi)
    g2 = gmag * np.sin(2.0 * phi)
    s7 = Sersic(y0, x0, n, peak=peak, r_e=r_e, g1=g1, g2=g2)
    print(s7(0.2, 2.1))

    # FWHM instead of r_e
    FWHM = 2 * r_e * (np.log(2.0) / Sersic.compute_kappa(n))**n
    s8 = Sersic(y0, x0, n, peak=peak, r_e=r_e, b_over_a=b_over_a, phi=phi)
    print(s8(0.2, 2.1))

    # flux instead of peak
    flux=Sersic.compute_flux(n, r_e, peak)
    s9 = Sersic(y0, x0, n, flux=flux, a=a, b=b, phi=phi)
    print(s9(0.2, 2.1))

    # make sure have access to r_e
    print('It is an error if following 9 or so values do not all match')
    print(s1.r_e)
    print(s2.r_e)
    print(s3.r_e)
    print(s4.r_e)
    print(s5.r_e)
    print(s6.r_e)
    print(s7.r_e)
    print(s8.r_e)
    print(s9.r_e)


    # make sure have access to a
    print('It is an error if following 9 or so values do not all match')
    print(s1.a)
    print(s2.a)
    print(s3.a)
    print(s4.a)
    print(s5.a)
    print(s6.a)
    print(s7.a)
    print(s8.a)
    print(s9.a)


    # make sure have access to b
    print('It is an error if following 9 or so values do not all match')
    print(s1.b)
    print(s2.b)
    print(s3.b)
    print(s4.b)
    print(s5.b)
    print(s6.b)
    print(s7.b)
    print(s8.b)
    print(s9.b)


    # make sure have access to phi
    print('It is an error if following 9 or so values do not all match')
    print(s1.phi)
    print(s2.phi)
    print(s3.phi)
    print(s4.phi)
    print(s5.phi)
    print(s6.phi)
    print(s7.phi)
    print(s8.phi)
    print(s9.phi)

    # make sure have access to C11
    print('It is an error if following 9 or so values do not all match')
    print(s1.C11)
    print(s2.C11)
    print(s3.C11)
    print(s4.C11)
    print(s5.C11)
    print(s6.C11)
    print(s7.C11)
    print(s8.C11)
    print(s9.C11)

    # make sure have access to C22
    print('It is an error if following 9 or so values do not all match')
    print(s1.C22)
    print(s2.C22)
    print(s3.C22)
    print(s4.C22)
    print(s5.C22)
    print(s6.C22)
    print(s7.C22)
    print(s8.C22)
    print(s9.C22)

    # make sure have access to C12
    print('It is an error if following 9 or so values do not all match')
    print(s1.C12)
    print(s2.C12)
    print(s3.C12)
    print(s4.C12)
    print(s5.C12)
    print(s6.C12)
    print(s7.C12)
    print(s8.C12)
    print(s9.C12)

if __name__ == '__main__':
    test_Sersic()
