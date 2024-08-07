import numpy as np
import pandas as pd
from gp_interpolation import GPInterpolation
import time
import h5py

lcs = pd.read_csv('../data/real/real_lcs_clean_90.csv')
meta = pd.read_csv('../data/real/real_meta_clean.csv')

ids_10 = meta.IAUID.unique()[:10]
print(ids_10)
lcs_10=lcs[lcs.IAUID.isin(ids_10)]
meta_10=meta[meta.IAUID.isin(ids_10)]


gpi = GPInterpolation(lcs_10,meta_10)

start_time = time.time()
num_chunks = 2
X, id = gpi.parallel_process(num_chunks)
end_time = time.time()
# Compute the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds with 2 chunks")

gpi.save_vectors('test_10_gps.h5')

with h5py.File('test_10_gps.h5','r') as f:
    X = f["X"]
    Y = f["Y"]
    ids = f["ids"]
    print(X.shape)
    print(len(Y))
    print(len(ids))