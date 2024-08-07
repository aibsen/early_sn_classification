import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import h5py

class GPInterpolation:
    def __init__(self, lcs, meta):
        self.ztf_central_wavelengths = {
                1: 4813.9, #g
                2: 6421.8, #r
            }

        self.class_dict = {
            'Ia':0,
            'Ib/c':1, 
            'II':2,
            'SLSN':3
        }
        
        self.lcs = lcs
        self.meta = meta
        self.nbands = 2
        self.length = 90
        
    def fit_gp(self,lc):
        times = lc.mjd
        magnitudes = lc.magpsf
        errors=lc.sigmapsf
        bands=lc.fid

        wavelengths=np.array([self.ztf_central_wavelengths[b] for b in bands])
        # Estimate initial scale using signal-to-noise ratio
        signal_to_noises = 1.0 / errors  # Assuming uncertainties in magnitudes
        initial_scale = np.abs(magnitudes[signal_to_noises.idxmax()])

        # Define Matern kernel for time
        matern_time = Matern(length_scale=20, nu=1.5,length_scale_bounds=(1e-2, 1e3))
        matern_wavelength = Matern(length_scale=1000, nu=1.5, length_scale_bounds=(1e-2, 1e9))

        # Combine kernels into a product kernel
        kernel = initial_scale**2 * matern_time * matern_wavelength

        gp = GaussianProcessRegressor(kernel=kernel, alpha=errors**2, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=20)

        gp.fit(np.vstack([times, wavelengths]).T, magnitudes)

        return gp

    def predict_gp(self, lc, gp):
        times = lc.mjd
        pred_times= np.arange(times.min(), times.max() + 1, 1)
        wavelengths_band1 = np.full_like(pred_times, self.ztf_central_wavelengths[1])
        wavelengths_band2 = np.full_like(pred_times, self.ztf_central_wavelengths[2])


        predictions_band1, prediction_variances_band1 = gp.predict(
            np.vstack([pred_times, np.full_like(pred_times, wavelengths_band1)]).T, return_std=True)
        predictions_band2, prediction_variances_band2 = gp.predict(
            np.vstack([pred_times, np.full_like(pred_times, wavelengths_band2)]).T, return_std=True)
        
        return (pred_times,predictions_band1, prediction_variances_band1,predictions_band2, prediction_variances_band2)
    
    def process_chunk(self, ids_chunk):
        chunk_df = self.lcs[self.lcs['IAUID'].isin(ids_chunk)]
        ids = []
        failed = []
        X = np.zeros((len(ids_chunk),self.nbands*2+1,self.length))
        
        for i,obj_id in enumerate(chunk_df['IAUID'].unique()):
            lc = chunk_df[chunk_df['IAUID'] == obj_id]
            try:
                gp = self.fit_gp(lc)
                predictions = self.predict_gp(lc,gp)
                lc_length = len(predictions[0])
                X[i,0,:lc_length] = predictions[1] #band1
                X[i,1,:lc_length] = predictions[2] #var band1
                X[i,2,:lc_length] = predictions[3] #band2
                X[i,3,:lc_length] = predictions[4] #var band2
                X[i,4,:lc_length] = predictions[0] #prediction times
                ids.append(obj_id)
                # print(gp.get_params())
            except Exception as e:
                failed.append((obj_id,e))
        return ids, X
    
    def map_class_to_number(self):
        types = self.meta[self.meta.IAUID.isin(self.ids)]['type']
        self.Y = [self.class_dict[t] for t in types]

    def parallel_process(self, num_chunks):
        # Get all unique IDs
        unique_ids = self.meta['IAUID'].unique()

        # Split the unique IDs into chunks
        id_chunks = np.array_split(unique_ids, num_chunks)

        # Use multiprocessing to process each chunk in parallel
        with Pool(processes=num_chunks) as pool:
            process_chunk_partial = partial(self.process_chunk)
            results = pool.map(process_chunk_partial, id_chunks)
            ids = [item for r in results for item in r[0]]
            X = np.concatenate([r[1] for r in results], axis=0)

        self.X = X
        self.ids = ids
        self.map_class_to_number()
        return X, ids
    
    def save_vectors(self, outputFile):
        hf=h5py.File(outputFile,'w')

        print("writing X")
        hf.create_dataset('X',data=self.X,compression="gzip", chunks=True, maxshape=(None,None,None,))

        print("writing ids")
        hf.create_dataset('ids',data=self.ids,compression="gzip", chunks=True, maxshape=(None,))
        
        print("writing Y")
        hf.create_dataset('Y',data=self.Y,compression="gzip", chunks=True, maxshape=(None,))

        hf.close()

    def plot(self,lc,prediction):
        predictions_band1 = prediction[0]
        non_zero_mask = predictions_band1 != 0
        predictions_band1_nonzero = predictions_band1[non_zero_mask]

        predictions_variances_band1 = prediction[1][non_zero_mask]
        predictions_band2 = prediction[2][non_zero_mask]
        predictions_variances_band2 = prediction[3][non_zero_mask]
        pred_times = prediction[4][non_zero_mask]

        lc_r=lc[lc.fid==2]
        lc_g=lc[lc.fid==1]
        plt.gca().invert_yaxis()
        plt.scatter(lc_r.mjd,lc_r.magpsf,color='red')
        plt.scatter(lc_g.mjd,lc_g.magpsf,color='green')
        plt.plot(pred_times,predictions_band2,color='red')
        plt.plot(pred_times,predictions_band1_nonzero,color='green')
        plt.fill_between(pred_times, predictions_band2 - predictions_variances_band2, predictions_band2 + predictions_variances_band2, color='red', alpha=0.2)
        plt.fill_between(pred_times, predictions_band1_nonzero - predictions_variances_band1, predictions_band1_nonzero + predictions_variances_band1, color='green', alpha=0.2)
        plt.show()