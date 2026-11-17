#%%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import h5py
import numpy as np
import numpy as np
import MAS_library as MASL
import Pk_library as PKL
from numpy.fft import fftn, fftshift

"""
Script to generate 3D power spectrum from snap directory.

Notes:
- CIC does not fully work as Pylians - needs revision.
- P(k) calculation affected by sample variance at low k modes (limited number of modes).
- Pylians computes the P(k) beyond Nyquist.
"""

def get_pk3d_from_snap(snapshot_dir, not_Pylians, save, nbins=1024, box_size=20):
    # Directory containing the snapshots
    # snapshot_dir = '/data/emergence10/prace_relics/planck1_20_1024/snapdir_012'
    # snapshot_dir = '/data/emergence13/og313/s8_project/DiRAC_relics_CWDM/planck1_20_1024/snapdir_000'


    # List all files in the directory
    snapshot_files = [f for f in os.listdir(snapshot_dir) if f.startswith('snap_') and f.endswith('.hdf5')]

    # Sort the files to ensure they are processed in order
    snapshot_files.sort()

    all_coordinates = []
    # Loop through each snapshot file
    for idx, snapshot_file in enumerate(snapshot_files):
        snapshot_path = os.path.join(snapshot_dir, snapshot_file)
        with h5py.File(snapshot_path, 'r') as f:
            if idx == 1:
                print('Redshift', f['Header'].attrs['Redshift'])
                redshift = round(f['Header'].attrs['Redshift'],2)
            pos_dm = f["PartType1"]["Coordinates"][:]
            all_coordinates.append(pos_dm) 
            print(f'File: {snapshot_file}, number of DM particles: {np.shape(pos_dm)}')

    # Concatenate all coordinates into a single numpy array
    all_coordinates = np.concatenate(all_coordinates, axis=0) / 1000


    # Print the shape of the final array
    print(f'Total DM particle positions: {np.shape(all_coordinates)}')
    print('Number of DM particles you would expect', nbins**3)

    # Initialize density grid
    density_grid = np.zeros((nbins, nbins, nbins), dtype=np.float32)

    # Normalize positions to grid coordinates
    grid_cell = box_size / nbins
    print('Grid cell size:', grid_cell)
    grid_coords = all_coordinates / grid_cell  # Now in grid units
    print('Grid coordinates:', grid_coords)
    # mass = 5.37e5 *(0.67)*2e30

    ## MAS with Pylians
    # define 3D density field
    grid = nbins
    pos = all_coordinates #/ 1000 # convert to Mpc/h
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    BoxSize = box_size #Mpc/h ; size of box
    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress

    # construct 3D density field
    MASL.MA(pos, delta, BoxSize, MAS, verbose=verbose)

    # at this point, delta contains the effective number of particles in each voxel
    # now compute overdensity and density constrast
    delta /= np.mean(delta, dtype=np.float64)


    print("Before subtracting 1:")
    print("max:", np.max(delta), "min:", np.min(delta), "mean:", np.mean(delta))

    delta -= 1.0

    print("After subtracting 1:")
    print("max:", np.max(delta), "min:", np.min(delta), "mean:", np.mean(delta))


    #P(k) calculation with Pylians
    axis = 0
    threads = 56
    verbose = True
    BoxSize = 20.0 #Mpc/h ; size of box
    # compute power spectrum
    Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)


    # 3D P(k)
    kpy       = Pk.k3D
    Pk0     = Pk.Pk[:,0] #monopole

    # 1D P(k)
    kpy_1D       = Pk.k1D
    Pk0_1D     = Pk.Pk1D  #monopole

    # Save result to load afterwards:
    if save == True:
        folder_out = snapshot_dir
        np.savetxt(folder_out+f'/pk3d_z{redshift}.dat',
                np.transpose([kpy, Pk0]))
        
        print('SAVED!')


    if not_Pylians == True:

        # MANUAL COMPUTATION
        #1. Window function:
        def MAS_correction(k, scheme='CIC'):
            if scheme == 'NGP':
                return np.sinc(k / np.pi)
            elif scheme == 'CIC':
                # return np.sinc(k / (2.0 * np.pi))**2
                return np.sinc(k )**2
            elif scheme == 'TSC':
                return (np.sinc(k / (2.0 * np.pi)))**3
            else:
                return 1.0  # no correction

        from numpy.fft import fftn, fftshift
        # def compute_power_spectrum_3D(delta, BoxSize):
        #     """
        #     Computes spherically averaged power spectrum P(k) from a 3D overdensity field delta(x).
        #     delta: 3D density contrast field (i.e., rho/mean - 1), shape (N, N, N)
        #     BoxSize: size of the box in Mpc/h
        #     Returns: k (1D array), P(k) (1D array)
        #     """
        N = delta.shape[0]
        assert delta.shape[1] == N and delta.shape[2] == N, "Must be a cube"

        print('Compute Fourier transform of delta(x)')
        delta_k = fftn(delta, norm = 'ortho')
        delta_k = fftshift(delta_k)


        # Compute k-vectors
        kvec = np.fft.fftfreq(N, d=BoxSize/N)*2*np.pi
        print(kvec)
        kvec = fftshift(kvec)
        print('kvec after shift:', kvec)
        kx, ky, kz = np.meshgrid(kvec, kvec, kvec, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)


        # MA correction cinlude here:
        print()
        # Apply the MAS correction in Fourier space
        kx_c = MAS_correction(np.pi * kx / N, scheme=MAS)
        print('kx_c:', kx_c)
        ky_c = kx_c
        kz_c = kx_c

        MAS_correction_total = kx_c * ky_c * kz_c
        print('MAS correction factor is:', MAS_correction_total)
        # Undo the smoothing introduced by the MAS
        delta_k /= MAS_correction_total

        # Compute power spectrum |delta_k|^2
        Pk3D = np.abs(delta_k)**2 


        # Flatten arrays
        k_flat = k_mag.ravel()
        P_flat = Pk3D.ravel()

        # Define k-bins
        k_min = np.min(k_flat[np.nonzero(k_flat)])
        k_max = np.max(k_flat)
        # k_bins = np.linspace(k_min, k_max, num=1000)
        bin_width = 0.01
        k_bins = np.arange(np.log10(k_min) - bin_width, np.log10(k_max) + bin_width, 2*bin_width)
        k_flat = np.log10(k_flat)

        k_inds = np.digitize(k_flat, k_bins)   # shift to 0-based
        print('Before shift', k_inds)
        k_inds = np.digitize(k_flat, k_bins) - 1  # shift to 0-based
        print('After shift', k_inds)
        print('Binning process')

        print('k_bins:', len(k_bins))

        # Mask out-of-range indices
        valid_mask = (k_inds >= 0) & (k_inds < len(k_bins)-1)

        # Vectorized binning
        Nmodes = np.bincount(k_inds[valid_mask], minlength=len(k_bins))
        Pk_sum = np.bincount(k_inds[valid_mask], weights=P_flat[valid_mask], minlength=len(k_bins))
        # Pk_sum = np.bincount(k_inds[valid_mask], minlength=len(k_bins))


        # Avoid division by zero
        nonzero = Nmodes > 0
        Pk = np.zeros_like(Pk_sum)
        Pk[nonzero] = Pk_sum[nonzero] / Nmodes[nonzero]

        # Bin centers
        k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

        # Only keep bins with modes and skip the first bin (k=0)
        valid = (Nmodes[1:] > 0)

    #Plot:
    plt.plot(kpy, Pk0,label='Pylians, 3D')  # Log-log scale
    plt.plot(kpy_1D, Pk0_1D,label='Pylians, 1D')  # Log-log scale
    if not_Pylians == True:
        plt.plot(10**k_centers[valid], Pk[1:][valid] * (20/1024)**3,label='Test code')  # Log-log scale   
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)  # Label for x-axis
    plt.ylabel(r'$P(k)$', fontsize=14)  # Label for y-axis
    plt.title(f'z = {redshift}', fontsize=16)  # Title with redshift
    plt.legend()
    # plt.savefig('/data/og313/sims/year2/22_05/pk_4.2_Wk_bef_equal_bins.pdf')
    plt.show()


#Run:
snapshot_dir = '/data/emergence13/og313/s8_project/DiRAC_relics_CWDM/planck1_20_1024/snapdir_007'
not_Pylians = False
save = True 
get_pk3d_from_snap(snapshot_dir, not_Pylians, save, nbins=1024, box_size=20)
# %%
