import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

correction_modes = [("noCorr"),    # no correction
                    ("Perturb")]   # perturbative correction

size_modes = [("FWHMHLR", r"(FWHM$_\mathrm{PSF}$ and HLR$_\mathrm{gal}$ fixed)"), # hold FWHM_psf and HLR_gal fixed
              ("r2r2", r"$r^2_\mathrm{PSF}$ and $r^2_\mathrm{gal}$ fixed"),       # hold r2_psf and r2_gal fixed
              ("FWHMFWHM", r"PSF and convolved gal FWHM fixed")]                  # hold FWHM_psf and (psf convolved with gal)_FWHM fixed

physics_modes = [("both", "DCR and chromatic seeing"),  # don"t turn anything off
                 ("DCR", "DCR only"),                   # turn off chromatic seeing,
                 ("CS", "Chromatic seeing only")]       # turn off DCR

profile_modes = [("GG", "Gaussian gal, Gaussian PSF", "blue"),   # Gaussian gal, Gaussian PSF
                 ("DG", "deV gal, Gaussian PSF", "red"),         # deV galaxy, Gaussian PSF
                 ("GM", "Gaussian gal, Moffat PSF", "magenta"),  # Gaussian galaxy, Moffat PSF
                 ("DM", "deV gal, Moffat PSF", "green")]         # deV galaxy, Moffat PSF

for size_mode in size_modes:
    for physics_mode in physics_modes:
        for correction_mode in correction_modes:

            fig = plt.figure(figsize=(7,5))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212, sharex=ax1)
            plt.setp(ax1.get_xticklabels(), visible=False)

            ax1.text(0.1, 0.11, physics_mode[1], fontsize=14)
            ax1.text(0.1, 0.09, size_mode[1], fontsize=14)

            ax1.set_ylabel('m')
            ax2.set_ylabel('c')
            ax2.set_xlabel("redshift")

            ax1.set_xlim(0.0, 2.0)
            ax1.set_ylim(-0.01, 0.13)

            ax2.set_xlim(0.0, 2.0)
            ax2.set_ylim(-0.01, 0.1)

            outfilename = "output/ring_vs_z_"
            outfilename += physics_mode[0]+'_'
            outfilename += correction_mode+'_'
            outfilename += size_mode[0]+".pdf"

            plotted = False
            for profile_mode in profile_modes:
                infilename = "output/ring_vs_z_"
                infilename += profile_mode[0]+'_'
                infilename += physics_mode[0]+'_'
                infilename += correction_mode+'_'
                infilename += size_mode[0]+".dat"
                try:
                    z, m1a, m1r, m2a, m2r, c1a, c1r, c2a, c2r = np.loadtxt(infilename).T
                except:
                    break
                plotted = True

                if size_mode[0] == "r2r2":
                    if profile_mode == profile_modes[0]:
                        ax1.plot(z, m1a, color='k', label="analytic")
                        ax1.plot(z, m2a, color='k')
                        ax2.plot(z, c1a, color='k', label="analytic")
                        ax2.plot(z, c2a, color='k')
                else:
                    ax1.plot(z, m1a, color=profile_mode[2])
                    ax1.plot(z, m2a, color=profile_mode[2])
                    ax2.plot(z, c1a, color=profile_mode[2])
                    ax2.plot(z, c2a, color=profile_mode[2])

                if correction_mode == 'Perturb':
                    m1r *= 10
                    m2r *= 10
                    c1r *= 10
                    c2r *= 10
                    ax2.text(0.05, 0.085, r"Curves: Analytic (no corrections)")
                    ax2.text(0.05, 0.072, r"Symbols: 10$\times$ ring test (with corrections)")

                ax1.scatter(z, m1r, color="None", marker='+',
                            edgecolor=profile_mode[2], label=profile_mode[1])
                ax1.scatter(z, m2r, color="None", marker='x', edgecolor=profile_mode[2])
                ax2.scatter(z, c1r, color="None", marker='+',
                            edgecolor=profile_mode[2], label=profile_mode[1])
                ax2.scatter(z, c2r, color="None", marker='x', edgecolor=profile_mode[2])

                ax1.legend(fontsize=9)
                ax2.legend(fontsize=9)

            if plotted:
                fig.tight_layout()
                fig.savefig(outfilename, dpi=220)
            else:
                plt.close(fig)
