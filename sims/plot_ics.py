#%%
#!/usr/bin/env python
"""
Script to compare power spectra ratios between simulations and theory predictions.

Loads power spectra from ICS files for different cosmological configurations
(CAMB, f_iso=1, f_iso=4) and plots their ratios to compare simulation outputs
against theoretical initial conditions.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Instead of plotting the ratios, change code to plot the dimensionless matter power spectra 

### READ DATA FROM ICS FILES

# CAMB initial conditions (reference baseline)
file_pattern_camb = '/data/emergence13/og313/planck1_20_1024/ics/ics.dat.*'
path_camb = f"{file_pattern_camb.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_camb, lgD_camb = np.loadtxt(path_camb, unpack=True)
interp_camb = interp1d(lgk_camb, lgD_camb, kind='linear')

# f_iso=4.0, n_iso=2
file_pattern_fiso4 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_4.0_niso_2/ics/ics.dat.*'
path_fiso4 = f"{file_pattern_fiso4.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso4, lgD_fiso4 = np.loadtxt(path_fiso4, unpack=True)
interp_fiso4 = interp1d(lgk_fiso4, lgD_fiso4, kind='linear')

file_pattern_fiso4sigma = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_4.0_niso_2_s8_0.969/ics/ics.dat.*'
path_fiso4sigma = f"{file_pattern_fiso4sigma.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso4sigma, lgD_fiso4sigma = np.loadtxt(path_fiso4sigma, unpack=True)
interp_fiso4sigma = interp1d(lgk_fiso4sigma, lgD_fiso4sigma, kind='linear')

# f_iso=1.0, n_iso=2
file_pattern_fiso1 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_1.0_niso_2/ics/ics.dat.*'
path_fiso1 = f"{file_pattern_fiso1.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso1, lgD_fiso1 = np.loadtxt(path_fiso1, unpack=True)
interp_fiso1 = interp1d(lgk_fiso1, lgD_fiso1, kind='linear')

# f_iso=2.5, n_iso=2
file_pattern_fiso25 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_2.5_niso_2/ics/ics.dat.*'
path_fiso25 = f"{file_pattern_fiso25.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso25, lgD_fiso25 = np.loadtxt(path_fiso25, unpack=True)
interp_fiso25 = interp1d(lgk_fiso25, lgD_fiso25, kind='linear')

# f_iso=2.0, n_iso=2
file_pattern_fiso20 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_2.0_niso_2/ics/ics.dat.*'
path_fiso20 = f"{file_pattern_fiso20.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso20, lgD_fiso20 = np.loadtxt(path_fiso20, unpack=True)
interp_fiso20 = interp1d(lgk_fiso20, lgD_fiso20, kind='linear')

# f_iso=7.0, n_iso=2
file_pattern_fiso70 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_7.0_niso_2/ics/ics.dat.*'
path_fiso70 = f"{file_pattern_fiso70.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso70, lgD_fiso70 = np.loadtxt(path_fiso70, unpack=True)
interp_fiso70 = interp1d(lgk_fiso70, lgD_fiso70, kind='linear')

file_pattern_fiso10 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_10.0_niso_2/ics/ics.dat.*'
path_fiso10 = f"{file_pattern_fiso10.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso10, lgD_fiso10 = np.loadtxt(path_fiso10, unpack=True)
interp_fiso10 = interp1d(lgk_fiso10, lgD_fiso10, kind='linear')


# f_iso=0.05, n_iso=3
file_pattern_fiso005 = '/data/emergence13/og313/DiRAC_relics_ISO/niso3/ics/planck1_20_1024_fiso_0.05_niso_3/ics.dat.*'
path_fiso005 = f"{file_pattern_fiso005.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso005, lgD_fiso005 = np.loadtxt(path_fiso005, unpack=True)
interp_fiso005 = interp1d(lgk_fiso005, lgD_fiso005, kind='linear')

file_pattern_fiso50='/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_50.0_niso_2/ics/ics.dat.*'
path_fiso50 = f"{file_pattern_fiso50.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso50, lgD_fiso50 = np.loadtxt(path_fiso50, unpack=True)
interp_fiso50 = interp1d(lgk_fiso50, lgD_fiso50, kind='linear')

file_pattern_fiso10sigma_zics = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_10.0_niso_2_s8_zics/ics/ics.dat.*'
path_fiso10sigma = f"{file_pattern_fiso10sigma_zics.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso10sigma_zics, lgD_fiso10sigma_zics = np.loadtxt(path_fiso10sigma, unpack=True)
interp_fiso10sigma_zics = interp1d(lgk_fiso10sigma_zics, lgD_fiso10sigma_zics, kind='linear')

file_pattern_fiso10sigma = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_10.0_niso_2_s8/ics/ics.dat.*'
path_fiso10sigma_s8 = f"{file_pattern_fiso10sigma.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso10sigma_s8, lgD_fiso10sigma_s8 = np.loadtxt(path_fiso10sigma_s8, unpack=True)
interp_fiso10sigma_s8 = interp1d(lgk_fiso10sigma_s8, lgD_fiso10sigma_s8, kind='linear')

file_pattern_fiso006 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_0.06_niso_3.5/ics/ics.dat.*'
path_fiso006 = f"{file_pattern_fiso006.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso006, lgD_fiso006 = np.loadtxt(path_fiso006, unpack=True)
interp_fiso006 = interp1d(lgk_fiso006, lgD_fiso006, kind='linear')

file_pattern_fiso18 = '/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_1.8_niso_2.5/ics/ics.dat.*'
path_fiso18 = f"{file_pattern_fiso18.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso18, lgD_fiso18 = np.loadtxt(path_fiso18, unpack=True)
interp_fiso18 = interp1d(lgk_fiso18, lgD_fiso18, kind='linear')

file_pattern_fiso03 = '/data/emergence13/og313/DiRAC_relics_ISO/niso3/planck1_20_1024_fiso_0.3_niso_3/ics/ics.dat.*'
path_fiso03 = f"{file_pattern_fiso03.split('ics.dat')[0]}pk_from_ics_dat/pk_ics.dat"
lgk_fiso03, lgD_fiso03 = np.loadtxt(path_fiso03, unpack=True)
interp_fiso03 = interp1d(lgk_fiso03, lgD_fiso03, kind='linear')

# Define common k-range for simulation data
logk_sim = np.linspace(
    max(lgk_camb[0], lgk_fiso4[0], lgk_fiso1[0], lgk_fiso25[0], lgk_fiso20[0], lgk_fiso70[0], lgk_fiso10[0], lgk_fiso50[0], lgk_fiso006[0], lgk_fiso18[0], lgk_fiso03[0]),#, lgk_fiso001[0], lgk_fiso005[0]),
    min(lgk_camb[-1], lgk_fiso4[-1], lgk_fiso1[-1], lgk_fiso25[-1], lgk_fiso20[-1], lgk_fiso70[-1], lgk_fiso10[-1], lgk_fiso50[-1], lgk_fiso006[-1], lgk_fiso18[-1], lgk_fiso03[-1]),#, lgk_fiso001[-1], lgk_fiso005[-1]),
    100
)
k_sim = 10**logk_sim

# Compute 3D power spectra using P(k) = (2π²/k³) × 10^(log P)
pk3d_camb = 10**interp_camb(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso4 = 10**interp_fiso4(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso1 = 10**interp_fiso1(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso25 = 10**interp_fiso25(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso20 = 10**interp_fiso20(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso70 = 10**interp_fiso70(logk_sim) * 2. * np.pi**2. / k_sim**3
pk3d_fiso10 = 10**interp_fiso10(logk_sim) * 2. * np.pi**2. / k_sim**3
pk3d_fiso006 = 10**interp_fiso006(logk_sim) * 2. * np.pi**2. / k_sim**3
pk3d_fiso18 = 10**interp_fiso18(logk_sim) * 2. * np.pi**2. / k_sim**3
pk3d_fiso03 = 10**interp_fiso03(logk_sim) * 2. * np.pi**2. / k_sim**3.
# pk3d_fiso015 = 10**interp_fiso015(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso005 = 10**interp_fiso005(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso50 = 10**interp_fiso50(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso4sigma = 10**interp_fiso4sigma(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso10sigma_s8 = 10**interp_fiso10sigma_s8(logk_sim) * 2. * np.pi**2. / k_sim**3.
pk3d_fiso10sigma_zics = 10**interp_fiso10sigma_zics(logk_sim) * 2. * np.pi**2. / k_sim**3.


### READ THEORY INPUT FILES 

# CAMB baseline theory
lgk_theory_camb, lgD_theory_camb = np.loadtxt(
    '/data/emergence8/viel/DiRAC_relics_CWDM/src/N-GenIC/planck1_matterpower_z=99_rightunits.dat',
    unpack=True
)
interp_theory_camb = interp1d(lgk_theory_camb, lgD_theory_camb, kind='linear')

# Theory predictions for f_iso=4, n_iso=2
lgk_theory_fiso4, lgD_theory_fiso4 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_4_niso_2.dat',
    unpack=True
)
interp_theory_fiso4 = interp1d(lgk_theory_fiso4, lgD_theory_fiso4, kind='linear')

# Theory predictions for f_iso=1, n_iso=2
lgk_theory_fiso1, lgD_theory_fiso1 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_1_niso_2.dat',
    unpack=True
)
interp_theory_fiso1 = interp1d(lgk_theory_fiso1, lgD_theory_fiso1, kind='linear')

# Theory predictions for f_iso=2.5, n_iso=2
lgk_theory_fiso25, lgD_theory_fiso25 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_2.5_niso_2.dat',
    unpack=True
)
interp_theory_fiso25 = interp1d(lgk_theory_fiso25, lgD_theory_fiso25, kind='linear')

# Theory predictions for f_iso=2.0, n_iso=2
lgk_theory_fiso20, lgD_theory_fiso20 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_2_niso_2.dat',
    unpack=True
)
interp_theory_fiso20 = interp1d(lgk_theory_fiso20, lgD_theory_fiso20, kind='linear')

# Theory predictions for f_iso=2.0, n_iso=2
lgk_theory_fiso7, lgD_theory_fiso7 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_7_niso_2.dat',
    unpack=True
)
interp_theory_fiso7 = interp1d(lgk_theory_fiso7, lgD_theory_fiso7, kind='linear')

lgk_theory_fiso10, lgD_theory_fiso10 = np.loadtxt(
    '/data/emergence13/og313/DiRAC_relics_ISO/sims/vid_code/ics_pk/pk_fiso_10_niso_2.dat',
    unpack=True
)

interp_theory_fiso10 = interp1d(lgk_theory_fiso10, lgD_theory_fiso10, kind='linear')

# Theory predictions for f_iso=2.0, n_iso=2
lgk_theory_fiso03, lgD_theory_fiso03 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_0.3_niso_3.dat',
    unpack=True
)
interp_theory_fiso03 = interp1d(lgk_theory_fiso03, lgD_theory_fiso03, kind='linear')

# Theory predictions for f_iso=2.0, n_iso=2
lgk_theory_fiso015, lgD_theory_fiso015 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_0.15_niso_3.dat',
    unpack=True
)
interp_theory_fiso015 = interp1d(lgk_theory_fiso015, lgD_theory_fiso015, kind='linear')

# Theory predictions for f_iso=2.0, n_iso=2
lgk_theory_fiso005, lgD_theory_fiso005 = np.loadtxt(
    '/data/og313/iso/vid_code/ics_pk/pk_fiso_0.05_niso_3.dat',
    unpack=True
)
interp_theory_fiso005 = interp1d(lgk_theory_fiso005, lgD_theory_fiso005, kind='linear')

lgk_theory_fiso50, lgD_theory_fiso50 = np.loadtxt(
    '/data/emergence13/og313/DiRAC_relics_ISO/sims/vid_code/ics_pk/pk_fiso_50_niso_2.dat',
    unpack=True
)
interp_theory_fiso50 = interp1d(lgk_theory_fiso50, lgD_theory_fiso50, kind='linear')

lgk_theory_fiso006, lgD_theory_fiso006 = np.loadtxt(
    '/data/emergence13/og313/DiRAC_relics_ISO/sims/vid_code/ics_pk/pk_fiso_0.06_niso_3.5_z_99.dat',
    unpack=True
)
interp_theory_fiso006 = interp1d(lgk_theory_fiso006, lgD_theory_fiso006, kind='linear')

lgk_theory_fiso18, lgD_theory_fiso18 = np.loadtxt(
    '/data/emergence13/og313/DiRAC_relics_ISO/sims/vid_code/ics_pk/pk_fiso_1.8_niso_2.5_z_99.dat',
    unpack=True
)
interp_theory_fiso18 = interp1d(lgk_theory_fiso18, lgD_theory_fiso18, kind='linear')

lgk_theory_fiso03, lgD_theory_fiso03 = np.loadtxt(
    '/data/emergence13/og313/DiRAC_relics_ISO/sims/vid_code/ics_pk/pk_fiso_0.3_niso_3.dat',
    unpack=True
)
interp_theory_fiso03 = interp1d(lgk_theory_fiso03, lgD_theory_fiso03, kind='linear')
# Define common k-range for theory data
logk_theory = np.linspace(
    max(lgk_theory_camb[0], lgk_theory_fiso4[0], lgk_theory_fiso1[0], lgk_theory_fiso25[0], lgk_theory_fiso20[0],  lgk_theory_fiso03[0], lgk_theory_fiso015[0], lgk_theory_fiso005[0], lgk_theory_fiso015[0], lgk_theory_fiso10[0], lgk_theory_fiso006[0], lgk_theory_fiso18[0]),
    min(lgk_theory_camb[-1], lgk_theory_fiso4[-1], lgk_theory_fiso1[-1], lgk_theory_fiso25[-1], lgk_theory_fiso20[-1], lgk_theory_fiso03[-1], lgk_theory_fiso015[-1], lgk_theory_fiso005[-1], lgk_theory_fiso015[-1], lgk_theory_fiso10[-1], lgk_theory_fiso006[-1], lgk_theory_fiso18[-1]),
    100
)
k_theory = 10**logk_theory

# Compute 3D power spectra for theory
pk3d_theory_camb = 10**interp_theory_camb(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso4 = 10**interp_theory_fiso4(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso1 = 10**interp_theory_fiso1(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso25 = 10**interp_theory_fiso25(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso20 = 10**interp_theory_fiso20(logk_theory) * 2. * np.pi**2. / k_theory**3.
# pk3d_theory_fiso03 = 10**interp_theory_fiso03(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso7 = 10**interp_theory_fiso7(logk_theory) * 2. * np.pi**2. / k_theory**3.
# pk3d_theory_fiso015 = 10**interp_theory_fiso015(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso005 = 10**interp_theory_fiso005(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso10 = 10**interp_theory_fiso10(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso50 = 10**interp_theory_fiso50(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso006 = 10**interp_theory_fiso006(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso18 = 10**interp_theory_fiso18(logk_theory) * 2. * np.pi**2. / k_theory**3.
pk3d_theory_fiso03 = 10**interp_theory_fiso03(logk_theory) * 2. * np.pi**2. / k_theory**3.
# pk3d_theory_fiso10 = pk3d_theory_fiso10 + pk3d_theory_camb


### Get sigma8 from model that we have not rescaled:
#sigma_z0_zics99 = sigma_z99 * D_plus  where Dplus = D(z=0)/D(z=z_ics)
sigma_z0_zics99 = 0.0195726 * 78.3742
sigma_z0_zics300 = 0.0121683 * 4641.38

# Compute 3D power spectra for theory
# pk3d_theory_camb = 10**interp_theory_camb(logk_theory) * 2. * np.pi**2. / k_theory**3.
# pk3d_theory_fiso4 = 10**interp_theory

# PLOTTING PARAMETERS


# Grid and box parameters for Nyquist frequency calculation
GRID_SIZE = 1024  # number of grid points
BOX_SIZE_MPC = 20  # box size in Mpc/h
NYQUIST_WAVENUMBER = GRID_SIZE * np.pi / BOX_SIZE_MPC

# Figure parameters
FIG_WIDTH = 8
FIG_HEIGHT = 6
FONT_SIZE = 16



plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.rcParams.update({'font.size': FONT_SIZE})

plt.plot(k_sim, pk3d_fiso4*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=4, n_{\rm{iso}}=2$ ICs', color='red')
plt.plot(k_theory, pk3d_theory_fiso4*(k_theory**3)/(2*np.pi**2), 
         color='red', linestyle='dashed', label=rf'Theory')

plt.plot(k_sim, pk3d_fiso4sigma*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=4, n_{\rm{iso}}=2$ ICs, $\sigma_8$', linestyle='dotted', color='darkred')

plt.plot(k_sim, pk3d_fiso006*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=0.06, n_{\rm{iso}}=3.5$ ICs', color='purple')
plt.plot(k_theory, pk3d_theory_fiso006*(k_theory**3)/(2*np.pi**2), 
         color='purple', linestyle='dashed', label=rf'Theory')

plt.plot(k_sim, pk3d_fiso18*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=1.8, n_{\rm{iso}}=2.5$ ICs', color='orange')
plt.plot(k_theory, pk3d_theory_fiso18*(k_theory**3)/(2*np.pi**2), 
         color='orange', linestyle='dashed', label=rf'Theory')

plt.plot(k_sim, pk3d_fiso10*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=10, n_{\rm{iso}}=2$ ICs', color='blue')

plt.plot(k_theory, pk3d_theory_fiso10*(k_theory**3)/(2*np.pi**2), 

         color='blue', linestyle='dashed', label=rf'Theory')

plt.plot(k_sim, pk3d_fiso03*(k_sim**3)/(2*np.pi**2),
         label=r'$f_{\rm{iso}}=0.3, n_{\rm{iso}}=3$ ICs', color='green')
plt.plot(k_theory, pk3d_theory_fiso03*(k_theory**3)/(2*np.pi**2), 
         color='green', linestyle='dashed', label=rf'Theory')

# CAMB default:
plt.plot(k_sim, pk3d_camb*(k_sim**3)/(2*np.pi**2),
         label=r'CAMB ICs', color='black')
plt.plot(k_theory, pk3d_theory_camb*(k_theory**3)/(2*np.pi**2), 
         color='black', linestyle='dashed', label=rf'Theory')

# Mark Nyquist frequency and region where resolution issues begin
plt.axvline(NYQUIST_WAVENUMBER, linestyle='--', color='black')
plt.axvspan(NYQUIST_WAVENUMBER, np.max(k_sim), color='gray', alpha=0.3, 
            label='Beyond Nyquist')

print('What is the Nyquist frequency in h/Mpc?', NYQUIST_WAVENUMBER)

plt.ylim(1e-4, 1)
plt.xlim(1, 400)

# Reference line at unity ratio
# plt.axhline(y=1, color='black', linestyle='--')

# Labels and formatting
plt.xlabel(r'$k \; \mathrm{[h/Mpc]}$')
plt.ylabel(fr'$\mathcal{{P}}(k)$')
plt.gca().set_xscale('log')
plt.xlim(2*np.pi/20, 500)
# plt.ylim(0.2, 150)
plt.yscale('log')
# Put legend outside the plot
plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1), borderaxespad=0.)

plt.show()


# %%
