import numpy
import scipy.integrate
import scipy.linalg

def linear_spec(V_AB, I_AB):
    V_data = numpy.genfromtxt('../../data/filters/WFPC2_F606W.dat')
    I_data = numpy.genfromtxt('../../data/filters/WFPC2_F814W.dat')
    V_wave, V_throughput = V_data[:,0], V_data[:,1]
    I_wave, I_throughput = I_data[:,0], I_data[:,1]

    AB_fnu = 1.0e-26
    speed_of_light = 2.99792458e18
    AB_wave = numpy.arange(200.0, 2000.0, 1.0)
    AB_flambda = AB_fnu * speed_of_light / AB_wave**2

    V_L = 470.0
    V_M = 595.0
    V_U = 750.0

    I_L = 690.0
    I_M = 833.5
    I_U = 1000.0

    f_V_AB = 10**(-0.4 * V_AB)
    f_I_AB = 10**(-0.4 * I_AB)

    # V-band integrals

    # first integral
    wave = V_wave[V_wave <= V_M]
    wave = wave[wave >= V_L]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    M1_V = scipy.integrate.simps(throughput * wave, wave)
    # second integral
    C1_V = M1_V #ha!
    # third integral
    wave = V_wave[V_wave <= V_U]
    wave = wave[wave >= V_M]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    M2_V = scipy.integrate.simps(throughput * wave * wave, wave)
    # fourth integral
    C2_V = scipy.integrate.simps(throughput * wave, wave)
    # denominator
    wave = V_wave[V_wave <= V_U]
    wave = wave[wave >= V_L]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    AB_flambda = AB_fnu * speed_of_light / wave**2
    den_V = scipy.integrate.simps(AB_flambda * throughput * wave, wave)

    # I-band integrals

    # first integral
    wave = I_wave[I_wave <= I_M]
    wave = wave[wave >= I_L]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    M1_I = scipy.integrate.simps(throughput * wave * wave, wave)
    # second integral
    C1_I = scipy.integrate.simps(throughput * wave, wave)
    # third integral
    wave = I_wave[I_wave <= I_U]
    wave = wave[wave >= I_M]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    M2_I = scipy.integrate.simps(throughput * wave, wave)
    # fourth integral
    C2_I = M2_I #ha!
    # denominator
    wave = I_wave[I_wave <= I_U]
    wave = wave[wave >= I_L]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    AB_flambda = AB_fnu * speed_of_light / wave**2
    den_I = scipy.integrate.simps(AB_flambda * throughput * wave, wave)

    A = numpy.array([[M1_V * V_M + M2_V, C1_V + C2_V],
                    [M1_I + M2_I * I_M, C1_I + C2_I]])
    return scipy.linalg.inv(A).dot(numpy.array([f_V_AB * den_V, f_I_AB * den_I]))


def AB_mags(m, c):
    V_data = numpy.genfromtxt('../../data/filters/WFPC2_F606W.dat')
    I_data = numpy.genfromtxt('../../data/filters/WFPC2_F814W.dat')
    V_wave, V_throughput = V_data[:,0], V_data[:,1]
    I_wave, I_throughput = I_data[:,0], I_data[:,1]

    AB_fnu = 1.0e-26
    speed_of_light = 2.99792458e18
    AB_wave = numpy.arange(200.0, 2000.0, 1.0)

    V_L = 470.0
    V_M = 595.0
    V_U = 750.0

    I_L = 690.0
    I_M = 833.5
    I_U = 1000.0

    #V-band

    # first integral
    wave = V_wave[V_wave >= V_L]
    wave = wave[wave <= V_M]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    S_flambda = m * V_M + c
    num1 = scipy.integrate.simps(S_flambda * throughput * wave, wave)
    # second integral
    wave = V_wave[V_wave >= V_M]
    wave = wave[wave <= V_U]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    S_flambda = m * wave + c
    num2 = scipy.integrate.simps(S_flambda * throughput * wave, wave)
    # third integral
    wave = V_wave[V_wave >= V_L]
    wave = wave[wave <= V_U]
    throughput = numpy.interp(wave, V_wave, V_throughput)
    AB_flambda = AB_fnu * speed_of_light / wave**2
    den = scipy.integrate.simps(AB_flambda * throughput * wave, wave)

    V_AB = -2.5 * numpy.log10((num1+num2)/den)

    # repeat for I-band

    # first integral
    wave = I_wave[I_wave >= I_L]
    wave = wave[wave <= I_M]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    S_flambda = m * wave + c
    num1 = scipy.integrate.simps(S_flambda * throughput * wave, wave)
    # second integral
    wave = I_wave[I_wave >= I_M]
    wave = wave[wave <= I_U]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    S_flambda = m * I_M + c
    num2 = scipy.integrate.simps(S_flambda * throughput * wave, wave)
    # third integral
    wave = I_wave[I_wave >= I_L]
    wave = wave[wave <= I_U]
    throughput = numpy.interp(wave, I_wave, I_throughput)
    AB_flambda = AB_fnu * speed_of_light / wave**2
    den = scipy.integrate.simps(AB_flambda * throughput * wave, wave)

    I_AB = -2.5 * numpy.log10((num1+num2)/den)

    return V_AB, I_AB
