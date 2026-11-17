PhD code/ bookkeeping
This repository contains 4 directories: branches:
1. nn_emulator: the code from nn_emulator_fwdm folder in /code_wdm/. The models saved with data used, input and labels files are in tar files in Google and saved locally.
2. plots_paper: this is the code from code_plots in /code_wdm/ that we mainly used to get plots for CWDM paper. The saved chains are in tar files in Google Drive and saved locally as well (run with myenv). We have also included scripts to plot projected density field, flux power spectra ratio, and T-rho relation.
3. mcmc: cobaya scripts running models from nn-emulator
4. sims: code to get ics from output N-GenIC files, plot theory versus computed particle power spectra, plot snapshot

Note, code to extract LOS, compute and post-process grids for flux power spectra calculation not included (personal version from collaboration internal code).