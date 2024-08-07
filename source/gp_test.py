import numpy as np
import pandas as pd
from gp_interpolation import GPInterpolation
import time

lcs = pd.read_csv('../data/real/real_lcs_clean_90.csv')
meta = pd.read_csv('../data/real/real_meta_clean.csv')

ids_10 = meta.IAUID.unique()[:10]
# print(ids_10)
lcs_10=lcs[lcs.IAUID.isin(ids_10)]
meta_10=meta[meta.IAUID.isin(ids_10)]


gpi = GPInterpolation(lcs_10,meta_10)

start_time = time.time()
num_chunks = 5
X,ids=gpi.parallel_process(num_chunks)
end_time = time.time()
# Compute the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds with 5 chunks")



# X = np.zeros(len(ids_10),4,80)
# Y = []
# ids = []
#n,ch,l
# print(ids)
for i,r in enumerate(ids):
    # print(r)
    lc = lcs_10[lcs_10.IAUID==r]
    prediction=X[i]
    gpi.plot(lc,prediction)
# gpi1 = GPInterpolation(lcs_10,meta_10)
# start_time = time.time()
# num_chunks = 1
# results1=gpi1.parallel_process(num_chunks)
# end_time = time.time()
# # Compute the elapsed time
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds with 1 chunk")

