# test.py

from model_loader import get_mixed_models
from config import redshift_configs
import numpy as np

z= 4.2
base_path='/data/og313/lyalpha/code_wdm/analysis_patchy'
# in_file=f'inputs{z}_patchy.txt'
# out_file=f'ps{z}_patchy.txt'


# Load models and normalization parameters for the redshift
models, min_in, max_in, taueff = get_mixed_models(z, base_path)#, in_file, out_file)

print(f"Loaded {len(models)} models for redshift {z}")
print(f"Min_in: {min_in}")
print(f"Max_in: {max_in}")
print(f"Tau_eff: {taueff}")

np.savetxt(f'{base_path}/norm_{z}.txt', [min_in, max_in])
