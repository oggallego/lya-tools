


from getdist import loadMCSamples

name_chain = 'chain_bo_WDM_pRELICS_u0T0_moC'


directory = f'/data/vault/og313/chains_irsic/{name_chain}'

smooth_2D = 0.5
smooth_1D = 0.5
ignore_rows = 0.8

    # Orden of names in Vid's saved chains


    # Load the samples without altering parameter names yet
    # To run this need .chainparamnames in the same folder
samples_irsic = loadMCSamples(directory, settings={
        'ignore_rows': ignore_rows,
        'smooth_scale_2D': smooth_2D,
        'smooth_scale_1D': smooth_1D
    })






stats = samples_irsic.getMargeStats()
names =  [
    'T0z42', 'T0z46', 'T0z5',
    'gammaz42', 'gammaz46', 'gammaz5',
    'taueffz42', 'taueffz46', 'taueffz5',
    'u0z42', 'u0z46', 'u0z5',
    'alpha' ]
original_names = ['taueffz42','taueffz46', 'taueffz5', 'T0z42', 'T0z46', 'T0z5',
                    'gammaz42', 'gammaz46', 'gammaz5', 
                    'u0z42', 'u0z46', 'u0z5',
                    'sigma8', 'neff', 'alpha', 'fqso']
names_ordered = ['T0z42' 'T0z46', 'T0z5', 'gammaz42', 'gammaz46', 'gammaz5', 'taueffz42', 'taueffz46', 'taueffz5', 'u0z42', 'u0z46', 'u0z5']
onesigma_all = []
for i in original_names:
    print(i)
    lims = stats.parWithName(f'{i}').limits
    for conf, lims in zip(samples_irsic.contours,lims):
            # (conf, lim0.lower, lim0.upper,
            # print(f'Parameter {i}: {conf}, {lims.lower} and {lims.upper}')
        if conf == 0.68:
            onesigma_all.append((lims.lower, lims.upper))
print(onesigma_all)
