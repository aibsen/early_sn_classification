import numpy as np
import pandas as pd
from gp_interpolation import GPInterpolation
import time
import h5py

lcs = pd.read_csv('../data/real/real_lcs_clean_90.csv')
meta = pd.read_csv('../data/real/real_meta_clean.csv')

gpi = GPInterpolation(lcs,meta)

start_time = time.time()
num_chunks = 10
gpi.parallel_process(num_chunks)
end_time = time.time()
# Compute the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds with 10 chunks")

gpi.save_vectors('real_data_90_days.h5')

