import numpy as np

'See test_boera.py for check of the final_covariance matrix'
def get_data():
    data42 = np.genfromtxt(f'/data/og313b/lyalpha/Boera_data/Boera19_z4.2', dtype=None, names=True)
    column_names42 = data42.dtype.names

    #Dictionary from the data:
    columns_as_arrays42 = {col: data42[col] for col in column_names42}
    log_k = columns_as_arrays42[column_names42[0]]
    P_kRC42_fid = columns_as_arrays42[column_names42[1]]
    P_kRC42 = columns_as_arrays42[column_names42[2]]

    data46 = np.genfromtxt(f'/data/og313b/lyalpha/Boera_data/Boera19_z4.6', dtype=None, names=True)
    column_names46 = data46.dtype.names
    columns_as_arrays46 = {col: data46[col] for col in column_names46}
    P_kRC46_fid = columns_as_arrays46[column_names46[1]]
    P_kRC46 = columns_as_arrays46[column_names46[2]]

    data5 = np.genfromtxt(f'/data/og313b/lyalpha/Boera_data/Boera19_z5', dtype=None, names=True)
    column_names5 = data5.dtype.names
    columns_as_arrays5 = {col: data5[col] for col in column_names5}
    P_kRC5_fid = columns_as_arrays5[column_names5[1]]
    P_kRC5 = columns_as_arrays5[column_names5[2]]

    #Power spectrum file:
    ps = np.genfromtxt('/data/og313b/lyalpha/Irsic2023/pk_mock_homog_zr53_r0.txt', dtype=float, delimiter=None)
    redshift = ps[:, 0]
    k = ps[:, 1] 



    # Load the covariance matrix and power spectrum data
    cov = np.genfromtxt('/data/og313b/lyalpha/Irsic2023/cov_mock_homog_zr53.txt', dtype=float, delimiter=None)
    rows = cov[:, 0].astype(int)
    columns = cov[:, 1].astype(int)
    elements = cov[:,2]
    matrix_size = max(np.max(rows), np.max(columns)) + 1
    cov_matrix = np.zeros((matrix_size, matrix_size))
    cov_matrix[rows, columns] = elements #(48,48) covariance matrix with all redshift info


    #Correct covariance matrix to use sigma from Boera:

    cov_matrix42= (np.load('/data/og313b/lyalpha/Boera19/anc/Cov_Matrixz=4.2.dat'))
    cov_matrix46= (np.load('/data/og313b/lyalpha/Boera19/anc/Cov_Matrixz=4.6.dat'))
    cov_matrix5= (np.load('/data/og313b/lyalpha/Boera19/anc/Cov_Matrixz=5.0.dat'))

    # print(cov_matrix42)

    #Window function eq. (7) in Boera19:
    R =  2.55 #km/s, nominal resolution #compared to R = 60 for paper: https://arxiv.org/pdf/1306.5896.pdf
    dv = 2.5 #km/s, pixel size

    def window_function(k):
        return np.exp(-((k*R)**2)/2)*((np.sin(k*dv/2))/(k*dv/2)) #eq. (7) in Boera19

    W = window_function(k)

    'Instead of taking the data, take my RC data values:'
    P_kRC42_new = P_kRC42_fid/(W[:16]**2)
    print('Our RC data', P_kRC42_new)
    P_kRC46_new = P_kRC46_fid/(W[16:32]**2)
    print('Our RC data', P_kRC46_new)
    P_kRC5_new = P_kRC5_fid/(W[32:]**2)
    print('Our RC data', P_kRC5_new)

    #Correct each element in the covariance matrix forinstrumental resolution (WINDOW FUNCTION):
    corrected_cov_matrix42 = cov_matrix42.copy()
    for i in range(cov_matrix42.shape[0]):
        for j in range(cov_matrix42.shape[1]):
            corrected_cov_matrix42[i,j] = corrected_cov_matrix42[i,j]/ (W[i] * W[j])
    # #print(corrected_cov_matrix42[11][3]) #Checked that outside of the loop it gives the same thing

    corrected_cov_matrix46 = cov_matrix46.copy()
    for i in range(cov_matrix46.shape[0]):
        for j in range(cov_matrix46.shape[1]):
            corrected_cov_matrix46[i,j] = corrected_cov_matrix46[i,j]/ (W[i] * W[j])

    corrected_cov_matrix5 = cov_matrix5.copy()
    for i in range(cov_matrix5.shape[0]):
        for j in range(cov_matrix5.shape[1]):
            corrected_cov_matrix5[i,j] = corrected_cov_matrix5[i,j]/ (W[i] * W[j])
            
    #cov_matrix_combined is the covariance matrix with three redshifts with the window function applied
    irsic_corr_matrix = np.zeros_like(cov_matrix, dtype=float)
    boera_corr_matrix = np.zeros_like(cov_matrix, dtype=float)
    covariance_matrix = np.zeros_like(cov_matrix, dtype=float)



    #Combined corrected matrices from Boera19:
    boera_cov_matrix_combined = np.concatenate((corrected_cov_matrix42, corrected_cov_matrix46, corrected_cov_matrix5), axis=1)
    corrected_cov_matrix = np.concatenate([boera_cov_matrix_combined] * 3, axis=0)

    for i in range(48):
        for j in range(48):
            #Correlation coefficient: r_ij = C_ij / sqrt(C_ii C_jj)
            #Diagonal elements should be 1
            if cov_matrix[i, i] == 0 or cov_matrix[j, j] == 0 or corrected_cov_matrix[i, i] == 0 or corrected_cov_matrix[j, j] == 0:
                print(f"Zero diagonal element at ({i}, {j})")
                continue

            #Get correlation coefficient from Irsic2023:
            irsic_corr_matrix[i, j] = cov_matrix[i, j] / np.sqrt(cov_matrix[i, i] * cov_matrix[j, j])
            #Get the correlation coefficient from corrected covariance matrix from Boera19:
            boera_corr_matrix[i, j] = corrected_cov_matrix[i, j] / np.sqrt(corrected_cov_matrix[i, i]*corrected_cov_matrix[j, j])
            covariance_matrix[i,j] = (corrected_cov_matrix[i,j]/boera_corr_matrix[i, j])*irsic_corr_matrix[i, j] 
    # print(covariance_matrix[:16, :16])
    return log_k, P_kRC42_new, P_kRC46_new, P_kRC5_new, covariance_matrix

log_k, P_kRC42_new, P_kRC46_new, P_kRC5_new, covariance_matrix = get_data()
np.savetxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/covariance_boera.txt', covariance_matrix)
np.savetxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC42_boera.txt', P_kRC42_new)
np.savetxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC46_boera.txt', P_kRC46_new)
np.savetxt('/data/og313/lyalpha/code_wdm/to_csd3/boera_data/P_kRC5_boera.txt', P_kRC5_new)