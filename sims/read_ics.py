#%%
import numpy as np
import glob
from tqdm import tqdm 
import Pk_library as PKL
import numpy as np
import MAS_library as MASL
import matplotlib.pyplot as plt
from numpy.fft import fftn, fftshift
import os 

# file_path = '/data/emergence9/og313/ics_test/ics.dat.0'
def read_ics(file_path, idx):
    """
    Read the initial conditions ics.dat file.
    Head structure can be inferred from allvars.h file in N-GenIC
    """

    pos1, vel1, id1 = [], [], []
    with open(file_path, 'rb') as f:

        header_dtype = np.dtype([
        ('npart', np.int32, 6),                # 24 bytes
        ('mass', np.float64, 6),               # 48 bytes
        ('time', np.float64),                  # 8
        ('redshift', np.float64),              # 8
        ('flag_sfr', np.int32),                # 4
        ('flag_feedback', np.int32),           # 4
        ('npartTotal', np.int32, 6),           # 24
        ('flag_cooling', np.int32),            # 4
        ('num_files', np.int32),               # 4
        ('BoxSize', np.float64),               # 8
        ('Omega0', np.float64),                # 8
        ('OmegaLambda', np.float64),           # 8
        ('HubbleParam', np.float64),           # 8
        ('flag_doubleprecision', np.int32),    # 4 ### NOT SURE IF THE ORDER OF THESE LAST 4 IS CORRECT!
        ('npartTotalHighWord', np.int32, 6),   # 24
        ('flag_stellarage', np.int32),         # 4
        ('flag_metals', np.int32),             # 4
        ('fill', 'S60'),                       # 60
        ])

        
        dummy1 = np.fromfile(f, dtype=np.int32, count=1)[0]
        header = np.fromfile(f, dtype=header_dtype, count=1)[0]
        dummy1b = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert dummy1 == dummy1b, "Header block size mismatch!"

        npart = header['npart']
        # total_particles = np.sum(npart)
   

        total_particles = np.sum(npart)


        # # Read all positions in one go
        flag_doubleprecision = header['flag_doubleprecision']
        dtype = np.float64 if flag_doubleprecision else np.float32

        # Read and skip the dummy before positions
        dummy2 = np.fromfile(f, dtype=np.int32, count=1)[0]

        
        positions = np.fromfile(f, dtype=dtype, count=total_particles * 3)
        # print(f"File {idx}: Read {len(positions)} floats")
        if len(positions) == total_particles * 3:
            positions = positions.reshape((total_particles, 3))
        # print('Positions', positions)

        dummy_pos_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        # Block B7: Read and skip the opening dummy for Velocities
        dummy_vel_start = np.fromfile(f, dtype=np.int32, count=1)[0]    
        # Test here if we cna also extract velocities:
        velocities = np.fromfile(f, dtype=dtype, count=total_particles * 3)
        # print(f"File {idx}: Read {len(positions)} floats")
        if len(velocities) == total_particles * 3:
            velocities = velocities.reshape((total_particles, 3))
        # print('Velocities', velocities)
        
        dummy_vel_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        dummy_id_start = np.fromfile(f, dtype=np.int32, count=1)[0]  

        id_dtype = np.int64
        expected_id_size = total_particles * 8
        if dummy_id_start < expected_id_size:
            id_dtype = np.int32
        
        particle_ids = np.fromfile(f, dtype=id_dtype, count=total_particles)
        # print('Particle IDs', particle_ids)
        dummy_id_end = np.fromfile(f, dtype=np.int32, count=1)[0]
        assert dummy_id_end == dummy_id_start, "ID block size mismatch!"

        print(f"Successfully read IDs using dtype: {id_dtype}")
   

        # # CHECK HEADER:
        # if idx == 0:
        #     # Check values
        #     print("npart:", header['npart'])  # Number of particles
        #     print("mass:", header['mass'])  # Mass of particles
        #     print("time:", header['time'])  # Time of the snapshot
        #     print("redshift:", header['redshift'])  # Redshift of the snapshot
        #     # print("flag_sfr:", header['flag_sfr'])  # Star formation flag
        #     # print("flag_feedback:", header['flag_feedback'])  # Feedback flag
        #     print("npartTotal:", header['npartTotal'])  # Total number of particles
        #     # print("npartTotalHighWord:", header['npartTotalHighWord'])  # High word for total number of particles
        #     # print("flag_cooling:", header['flag_cooling'])  # Cooling flag
        #     # print("flag_stellarage:", header['flag_stellarage'])  # Stellar age flag
        #     # print("flag_metals:", header['flag_metals'])  # Metals flag
        #     print("num_files:", header['num_files'])  # Number of files in the snapshot
        #     print("BoxSize:", header['BoxSize'])  # Size of the simulation box
        #     print("Omega0 (Matter Density):", header['Omega0'])  # Matter density parameter
        #     print("OmegaLambda (Dark Energy Density):", header['OmegaLambda'])  # Dark energy density parameter
        #     print("HubbleParam:", header['HubbleParam'])  # Hubble parameter
        #     # print("flag_doubleprecision:", header['flag_doubleprecision'])  # Double precision flag
        #     # print("fill (Padding):", header['fill'])  # Padding field

            
        


            # print("Get grid from total particles:", round(header['npartTotal'][1]**(1/3)))
            # print('-------------------------------')
            # pos1, vel1, id1 = positions[0], velocities[0], particle_ids[0]
            # print('Position, velocity, id', pos1, vel1, id1)

    return positions, header['BoxSize']/1000, round(header['npartTotal'][1]**(1/3))

# Function if we want to manually calculate Pk:

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



#### MAIN function:
def plot_pk_ics(file_pattern, pylians, save, name_out):

    files = sorted(glob.glob(file_pattern))  # Get all matching files and sort them

    all_positions = []

    for idx, file_path in enumerate(tqdm(files, desc="Processing files")):
        # print(f"Processing file: {file_path.split('ics.dat.')[1]}")
        positions, BoxSize, grid = read_ics(file_path, idx)
        print('Grid', grid)
        print('Boxsize', BoxSize)
        all_positions.append(positions)
    # print('Shape of all_positions before reshaping is', np.shape(all_positions))
    # Reshape while keeping the last axis as 3
    #all_positions = np.array(all_positions).reshape((-1, 3)) / 1000  # Convert to Mpc/h
    all_positions = np.concatenate(all_positions, axis=0) / 1000

    print('Shape of all_positions is', np.shape(all_positions))

    print('Using Pylians for MA!')
    pos = all_positions.astype(np.float32) #*BoxSize
    print('Grid',grid)
    print('BoxSize', BoxSize)

    MAS     = 'CIC'  #mass-assigment scheme
    verbose = True   #print information on progress

    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    MASL.MA(pos,  delta, BoxSize, MAS, verbose=verbose)
    print('MA completed')
    print('Computing density contrast:')
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0

    # For debugging:
    # print("pos min:", np.min(pos), "max:", np.max(pos))
    # print("BoxSize:", BoxSize)
    # print("delta min:", np.min(delta), "max:", np.max(delta), "mean:", np.mean(delta))
    # print("Unique values in slice:", np.unique(delta[:, :, grid//2]))
    # print("Any NaNs in delta?", np.isnan(delta).any())
    # print("Any infs in delta?", np.isinf(delta).any())

    # 3D P(k)
    if pylians == True:
        print('Calculating power spectrum with Pylians...')
        axis = 0 
        threads = 16
        Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads, verbose)
        kpy       = Pk.k3D
        Pk0     = Pk.Pk[:,0] #monopole
        plt.figure(figsize=(8, 6))
        plt.plot(kpy, Pk0, label='From ics.dat')  # Log-log scale
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)  # Label for x-axis
        plt.ylabel(r'$P(k)$', fontsize=14)  # Label for y-axis

        nyq_value = grid * np.pi / BoxSize
        plt.axvline(nyq_value, linestyle='--', color='black')  # Vertical line
        plt.axvspan(nyq_value, np.max(kpy), color='gray', alpha=0.3, label='Beyond Nyquist')
        plt.legend()
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        # plt.savefig('/data/og313/sims/planck1_5_256/ics/pk_ics.dat.pdf')
    
        plt.show()

        if save == True:
            folderout = f"{file_pattern.split('ics.dat')[0]}pk_from_ics_dat/"
            # folderout = f"/data/emergence8/og313/DiRAC_relics/planck1_20_1024/ics/pk_from_ics_dat/"
            os.makedirs(folderout, exist_ok=True)
            # Save in the same units as in Jamie's
            np.savetxt(folderout+f'{name_out}',
                    np.transpose([np.log10(kpy), np.log10(Pk0*kpy**3./(2.*np.pi**2.))]))
            print('Pk saved')

    else:
        print('Calculating power spectrum manually...')
        N = delta.shape[0]
        assert delta.shape[1] == N and delta.shape[2] == N, "Must be a cube"

        print('Compute Fourier transform of delta(x)')
        delta_k = fftn(delta, norm = 'ortho')
        delta_k = fftshift(delta_k)

        kvec = np.fft.fftfreq(N, d=BoxSize/N)*2*np.pi
        # print(kvec)
        kvec = fftshift(kvec)
        # print('kvec after shift:', kvec)
        kx, ky, kz = np.meshgrid(kvec, kvec, kvec, indexing='ij')
        k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

        kx_c = MAS_correction(np.pi * kx / N, scheme=MAS)
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
        # print('Before shift', k_inds)
        k_inds = np.digitize(k_flat, k_bins) - 1  # shift to 0-based
        # print('After shift', k_inds)
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


        plt.plot(10**k_centers[valid], Pk[1:][valid] * (BoxSize/grid)**3,label='Test code')
        plt.plot(kpy, Pk0,label='Pylians')  # Log-log scale
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.title(('z =99'))
        plt.xlabel(r'$k$ (h/Mpc)', fontsize=14)  # Label for x-axis
        plt.ylabel(r'$P(k)$', fontsize=14)  # Label for y-axis
        plt.legend()
        #Plot Nyquist frequency:
        # nyq_value = grid * np.pi / box
        # plt.axvline(nyq_value, linestyle='--', color='black')  # Vertical line
        # plt.axvspan(nyq_value, np.max(k_dm), color='gray', alpha=0.3, label='Beyond Nyquist')
        # plt.savefig('/data/og313/sims/year2/22_05/pk_4.2_Wk_bef_equal_bins.pdf')
        plt.show()




'New n_iso runs'
file_pattern = "/data/emergence13/og313/DiRAC_relics_ISO/planck1_20_1024_fiso_0.8_niso_3/ics/ics.dat.*"
name_out = 'pk_ics.dat'

plot_pk_ics(file_pattern, pylians=True, save=True, name_out=name_out)

'Then use plot_ics.py to plot the Pk againts CAMB or Jamies file'

# %%


# %%
