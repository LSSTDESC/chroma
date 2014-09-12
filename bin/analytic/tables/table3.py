# order is DES, LSST
r2galsqr = [0.6**2, 0.4**2]
r2PSFsqr = [0.8**2, 0.7**2]

m_req = [0.008, 0.003]
cRMS_req = [0.0025, 0.0015]

shear_var = 4.e-4

def MeanDeltaV(m, r2galsqr):
    return m * r2galsqr

def RMSDeltaV(c, r2galsqr):
    return c * r2galsqr * 2

def MeanDeltaRbar(m, r2galsqr):




outfile = open("chromatic_requirements.tex", 'w')
outfile.write("\\begin{deluxetable*}{lcccccc}\n")
outfile.write("  \\tablecaption{\label{table:chromatic_requirements} Requirements on chromatic bias parameters. }\n")
outfile.write("  \\tablehead{\n")
outfile.write("    \colhead{Survey} &\n")
outfile.write("    \colhead{$|\langle\Delta \\bar{R}\\rangle|$} &\n")
outfile.write("    \colhead{$(\Delta \\bar{R})_\mathrm{RMS}$} &\n")
outfile.write("    \colhead{$|\langle\Delta V\\rangle|$} &\n")
outfile.write("    \colhead{$(\Delta V)_\mathrm{RMS}$} &\n")
outfile.write("    \colhead{$|\langle\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf}\\rangle|$} &\n")
outfile.write("    \colhead{$(\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf})_\mathrm{RMS}$}\n")
outfile.write("  }\n")
outfile.write("  \startdata\n")
DESreqs = []
outfile.write("    DES  & {:5.3f} & {:5.3f} & {:5.3f} & {:6.4f} & {:5.3f} & {5.3f}")
outfile.write("    DES  & 0.045 & 0.036 & 0.002  & 0.0013 & 0.003  & 0.065 \\\n")
outfile.write("    LSST & 0.02  & 0.02  & 0.0004 & 0.0004 & 0.0008 & 0.016\n")
outfile.write("  \enddata\n")
outfile.write("  \\tablecomments{Requirements on bias parameters for differential chromatic refraction ($\Delta \\bar{R}$ and $\Delta V$) and for chromatic seeing ($\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf}$), defined in Equations~\\ref{eqn:Rbar}, \\ref{eqn:V}, and \\ref{eqn:delta_r2}.  Units are in arcseconds for $\Delta R$ and square arcseconds for $\Delta V$.\n")
outfile.write("  These are the values for which each bias by itself would degrade the accuracy and/or precision of the survey by an amount equivalent to the statistical sensitivity.}\n")
outfile.write("\end{deluxetable*}\n")
