import numpy as np

# order is DES, LSST
r2sqr_gal = np.r_[0.6, 0.4]**2
r2sqr_PSF = np.r_[0.8, 0.7]**2

mean_m_req = np.r_[0.008, 0.003]
shear_var = 4.e-4
var_c_req = mean_m_req * 2 * shear_var
print var_c_req

psf_ellip = 0.05

mean_DeltaRbarSqr_req = mean_m_req * 1.0
var_DeltaRbarSqr_req = var_c_req * 0.5**2

mean_DeltaV_req = r2sqr_gal * mean_m_req
var_DeltaV_req = r2sqr_gal**2 * var_c_req

mean_dr_by_dr_req = mean_m_req * r2sqr_gal / r2sqr_PSF
var_dr_by_dr_req = var_c_req * (r2sqr_gal / r2sqr_PSF * 2.0 / psf_ellip)**2

DESreqs = [a[0] for a in [mean_DeltaRbarSqr_req,
                          var_DeltaRbarSqr_req,
                          mean_DeltaV_req,
                          var_DeltaV_req,
                          mean_dr_by_dr_req,
                          var_dr_by_dr_req]]
LSSTreqs = [a[1] for a in [mean_DeltaRbarSqr_req,
                           var_DeltaRbarSqr_req,
                           mean_DeltaV_req,
                           var_DeltaV_req,
                           mean_dr_by_dr_req,
                           var_dr_by_dr_req]]

outfile = open("chromatic_requirements.tex", 'w')
outfile.write("\\begin{deluxetable*}{lcccccc}\n")
outfile.write("  \\tablecaption{\label{table:chromatic_requirements} Requirements on chromatic bias parameters. }\n")
outfile.write("  \\tablehead{\n")
outfile.write("    \colhead{Survey} &\n")
outfile.write("    \colhead{$\langle(\Delta \\bar{R}_{45})^2\\rangle_\mathrm{SEDs}$} &\n")
outfile.write("    \colhead{$\mathrm{Var}((\Delta \\bar{R}_{45})^2)_\mathrm{SEDs}$} &\n")
outfile.write("    \colhead{$|\langle\Delta V\\rangle_\mathrm{SEDs}|$} &\n")
outfile.write("    \colhead{$\mathrm{Var}(\Delta V)_\mathrm{SEDs}$} &\n")
outfile.write("    \colhead{$|\langle\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf}\\rangle_\mathrm{SEDs}|$} &\n")
outfile.write("    \colhead{$\mathrm{Var}(\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf})_\mathrm{SEDs}$}\n")
outfile.write("  }\n")
outfile.write("  \startdata\n")
outfile.write("    DES  & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ \\\\\n".format(*DESreqs))
outfile.write("    LSST & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$ & ${:.4e}$\n".format(*LSSTreqs))
outfile.write("  \enddata\n")
outfile.write("  \\tablecomments{Requirements on bias parameters for differential chromatic refraction ($\Delta \\bar{R}$ and $\Delta V$) and for chromatic seeing ($\Delta r^2_\mathrm{psf}/r^2_\mathrm{psf}$), defined in Equations~\\ref{eqn:Rbar}, \\ref{eqn:V}, and \\ref{eqn:delta_r2}.  Units are in arcseconds for $\Delta R$ and square arcseconds for $\Delta V$.\n")
outfile.write("  These are the values for which each bias by itself would degrade the accuracy and/or precision of the survey by an amount equivalent to the statistical sensitivity.}\n")
outfile.write("\\end{deluxetable*}\n")
