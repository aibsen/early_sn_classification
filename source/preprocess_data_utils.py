
from re import L
import math
import pandas as pd
import numpy as np
import h5py
import george
from typing import Dict, List
from functools import partial
from astropy.table import Table, vstack
import scipy.optimize as op


#modified from https://github.com/tallamjr/astronet/blob/master/astronet/preprocess.py
def predict_2d_gp(gp_predict, gp_times, gp_wavelengths):
    """Outputs the predictions of a Gaussian Process.
    Parameters
    ----------
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    gp_times : numpy.ndarray
        Times to evaluate the Gaussian Process at.
    gp_wavelengths : numpy.ndarray
        Wavelengths to evaluate the Gaussian Process at.
    Returns
    -------
    obj_gps : pandas.core.frame.DataFrame, optional
        Time, flux and flux error of the fitted Gaussian Process.
    Examples
    --------
    >>> gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    >>> number_gp = timesteps
    >>> gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    >>> obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    >>> obj_gps["filter"] = obj_gps["filter"].map(inverse_pb_wavelengths)
    ...
    """
    unique_wavelengths = np.unique(gp_wavelengths)
    number_gp = len(gp_times)
    obj_gps = []
    for wavelength in unique_wavelengths:
        gp_wavelengths = np.ones(number_gp) * wavelength
        pred_x_data = np.vstack([gp_times, gp_wavelengths]).T
        pb_pred, pb_pred_var = gp_predict(pred_x_data, return_var=True)
        # stack the GP results in a array momentarily
        obj_gp_pb_array = np.column_stack((gp_times, pb_pred, np.sqrt(pb_pred_var)))
        obj_gp_pb = Table(
            [
                obj_gp_pb_array[:, 0],
                obj_gp_pb_array[:, 1],
                obj_gp_pb_array[:, 2],
                [wavelength] * number_gp,
            ],
            names=["mjd", "flux", "flux_err", "passband"],
        )
        if len(obj_gps) == 0:  # initialize the table for 1st passband
            obj_gps = obj_gp_pb
        else:  # add more entries to the table
            obj_gps = vstack((obj_gps, obj_gp_pb))

    obj_gps = obj_gps.to_pandas()
    return obj_gps


def fit_2d_gp(
    obj_data: pd.DataFrame,
    return_kernel: bool = False,
    pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    **kwargs,
):
    """Fit a 2D Gaussian process.
    If required, predict the GP at evenly spaced points along a light curve.
    Parameters
    ----------
    obj_data : pd.DataFrame
        Time, flux and flux error of the data (specific filter of an object).
    return_kernel : bool, default = False
        Whether to return the used kernel.
    pb_wavelengths: dict
        Mapping of the passband wavelengths for each filter used.
    kwargs : dict
        Additional keyword arguments that are ignored at the moment. We allow
        additional keyword arguments so that the various functions that
        call this one can be called with the same arguments.
    Returns
    -------
    kernel: george.gp.GP.kernel, optional
        The kernel used to fit the GP.
    gp_predict : functools.partial of george.gp.GP
        The GP instance that was used to fit the object.
    Examples
    --------
    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)
    ...
    """
    guess_length_scale = 20.0  # a parameter of the Matern32Kernel

    obj_times = obj_data.mjd.astype(float)
    obj_flux = obj_data.flux.astype(float)
    obj_flux_error = obj_data.flux_err.astype(float)
    obj_wavelengths = obj_data["passband"].astype(str).map(pb_wavelengths)

    def neg_log_like(p):  # Objective function: negative log-likelihood
        gp.set_parameter_vector(p)
        loglike = gp.log_likelihood(obj_flux, quiet=True)
        return -loglike if np.isfinite(loglike) else 1e25

    def grad_neg_log_like(p):  # Gradient of the objective function.
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(obj_flux, quiet=True)

    # Use the highest signal-to-noise observation to estimate the scale. We
    # include an error floor so that in the case of very high
    # signal-to-noise observations we pick the maximum flux value.
    signal_to_noises = np.abs(obj_flux) / np.sqrt(
        obj_flux_error**2 + (1e-2 * np.max(obj_flux)) ** 2
    )
    scale = np.abs(obj_flux[signal_to_noises.idxmax()])

    kernel = (0.5 * scale) ** 2 * george.kernels.Matern32Kernel(
        [guess_length_scale**2, 6000**2], ndim=2
    )
    kernel.freeze_parameter("k2:metric:log_M_1_1")

    gp = george.GP(kernel)
    default_gp_param = gp.get_parameter_vector()
    x_data = np.vstack([obj_times, obj_wavelengths]).T
    gp.compute(x_data, obj_flux_error)

    bounds = [(0, np.log(1000**2))]
    bounds = [(default_gp_param[0] - 10, default_gp_param[0] + 10)] + bounds
    results = op.minimize(
        neg_log_like,
        gp.get_parameter_vector(),
        jac=grad_neg_log_like,
        method="L-BFGS-B",
        bounds=bounds,
        tol=1e-6,
    )

    if results.success:
        gp.set_parameter_vector(results.x)
    else:
        # Fit failed. Print out a warning, and use the initial guesses for fit
        # parameters.
        obj = obj_data["object_id"][0]
        print("GP fit failed for {}! Using guessed GP parameters.".format(obj))
        gp.set_parameter_vector(default_gp_param)

    gp_predict = partial(gp.predict, obj_flux)

    if return_kernel:
        return kernel, gp_predict
    return gp_predict



def generate_gp_single_event(
    df: pd.DataFrame, timesteps: int = 100, pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    var_length = False
) -> pd.DataFrame:
    """Intermediate helper function useful for visualisation of the original data with the mean of
    the Gaussian Process interpolation as well as the uncertainity.
    Additional steps required to build full dataframe for classification found in
    `generate_gp_all_objects`, namely:
        ...
        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="filter", values="flux")
        obj_gps = obj_gps.reset_index()
        obj_gps["object_id"] = object_id
        ...
    To allow a transformation from:
        mjd	        flux	    flux_error	filter
    0	0.000000	19.109279	0.176179	1(ztfg)
    1	0.282785	19.111843	0.173419	1(ztfg)
    2	0.565571	19.114406	0.170670	1(ztfg)
    to ...
    filter	mjd	        ztfg    ztfr	object_id
    0	    0	        19.1093	19.2713	27955532126447639664866058596
    1	    0.282785	19.1118	19.2723	27955532126447639664866058596
    2	    0.565571	19.1144	19.2733	27955532126447639664866058596
    Examples
    --------
    obj_gps = generate_gp_single_event(data)
    ax = plot_event_data_with_model(data, obj_model=_obj_gps, pb_colors=ZTF_PB_COLORS)
    """

    filters = df["passband"].astype(str)
    filters = list(np.unique(filters))

    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}

    gp_predict = fit_2d_gp(df, pb_wavelengths=pb_wavelengths)

    if var_length:
        mjd_diff = max(df['mjd'])-min(df['mjd'])
        number_gp = timesteps if mjd_diff>timesteps else int(np.floor(mjd_diff))
    else:
        number_gp = timesteps
    gp_times = np.linspace(min(df["mjd"]), max(df["mjd"]), number_gp)
    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    obj_gps["passband"] = obj_gps["passband"].map(inverse_pb_wavelengths)
    obj_gps["passband"] = obj_gps["passband"].astype(int)

    return obj_gps, number_gp

def create_gp_interpolated_vectors(
    object_list: List[str],
    obs_transient: pd.DataFrame,
    obs_metadata: pd.DataFrame,
    timesteps: int = 100,
    pb_wavelengths: Dict = ZTF_PB_WAVELENGTHS,
    var_length = False
) -> pd.DataFrame:
    """Generate Gaussian Process interpolation for all objects within 'object_list'. Upon
    completion, a dataframe is returned containing a value for each time step across each passband.
    Parameters
    ----------
    object_list: List[str]
        List of objects to apply the transformation to
    obs_transient: pd.DataFrame
        Dataframe containing observational points with the transient section of the full light curve
    timesteps: int
        Number of points one would like to interpolate, i.e. how many points along the time axis
        should the Gaussian Process be evaluated
    pb_wavelengths: Dict
        A mapping of passbands and the associated wavelengths, specific to each survey. Current
        options are ZTF or LSST
    Returns
    -------
    df: pd.DataFrame(data=adf, columns=obj_gps.columns)
        Dataframe with the mean of the GP for N x timesteps
    Examples
    --------
    ?>>> object_list = list(np.unique(df["object_id"]))
    ?>>> obs_transient, object_list = __transient_trim(object_list, df)
    ?>>> generated_gp_dataset = generate_gp_all_objects(
        object_list, obs_transient, timesteps, LSST_PB_WAVELENGTHS
        )
    ...
    """

    filters = obs_transient["passband"]
    filters = list(np.unique(filters))

    columns = []
    columns.append("mjd")
    # for filt in filters:
    #     columns.append(filt)
    columns.append("object_id")
    columns.append("passband")

    adf = pd.DataFrame(
        data=[],
        columns=columns,
    )
    id_list = []
    targets = []
    n_lcs = len(object_list)
    n_channels = 2
    X = np.ones((n_lcs, n_channels, timesteps)) #if flux is negative, set it to 1, so it can be converted to mag
    lens = np.zeros((n_lcs,))
    #if flux_err is negative, make it positive
    for i,object_id in enumerate(object_list):
        print(f"OBJECT ID:{object_id} at INDEX:{object_list.index(object_id)}")
        df = obs_transient[obs_transient["object_id"] == object_id]

        obj_gps, lc_length = generate_gp_single_event(df, timesteps, pb_wavelengths, var_length=var_length)
        # print(obj_gps)

        obj_gps = pd.pivot_table(obj_gps, index="mjd", columns="passband", values="flux")
        X[i,0,-lc_length:] = obj_gps[0]
        X[i,1,-lc_length:] = obj_gps[1]
        id_list.append(object_id)
        true_target = obs_metadata[obs_metadata.object_id==object_id].true_target.values[0]
        targets.append(true_target)
        lens[i] = lc_length
    #     obj_gps = obj_gps.reset_index()
        # obj_gps["object_id"] = object_id
        # adf = np.vstack((adf, obj_gps))
    
    X = np.where(X>0,X,1)
    # print(X)
    # print(X)
    return X, id_list, targets, lens
    # return pd.DataFrame(data=obj_gps, columns=obj_gps.columns)

def append_vectors(dataset,outputFile):
    with h5py.File(outputFile, 'a') as hf:
        X=dataset["X"]
        hf["X"].resize((hf["X"].shape[0] + X.shape[0]), axis = 0)
        hf["X"][-X.shape[0]:] = X

        ids = dataset["ids"]
        hf["ids"].resize((hf["ids"].shape[0] + ids.shape[0]), axis = 0)
        hf["ids"][-ids.shape[0]:] = ids

        Y=dataset["Y"]
        hf["Y"].resize((hf["Y"].shape[0] + Y.shape[0]), axis = 0)
        hf["Y"][-Y.shape[0]:] = Y
        hf.close()


def save_vectors(dataset, outputFile):
    hf=h5py.File(outputFile,'w')

    print("writing X")
    hf.create_dataset('X',data=dataset['X'],compression="gzip", chunks=True, maxshape=(None,None,None,))

    print("writing ids")
    hf.create_dataset('ids',data=dataset['ids'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    print("writing Y")
    hf.create_dataset('Y',data=dataset['Y'],compression="gzip", chunks=True, maxshape=(None,))

    if 'lens' in dataset.keys():
        print("writing lens")
        hf.create_dataset('lens',data=dataset['lens'],dtype='int64',compression="gzip", chunks=True, maxshape=(None,))
    
    hf.close()


    def fit_gaussian_process(
        self,
        fix_scale=False,
        verbose=False,
        guess_length_scale=20.0,
        **preprocessing_kwargs
    ):
        """Fit a Gaussian Process model to the light curve.

        We use a 2-dimensional Matern kernel to model the transient. The kernel
        width in the wavelength direction is fixed. We fit for the kernel width
        in the time direction as different transients evolve on very different
        time scales.

        Parameters
        ----------
        fix_scale : bool (optional)
            If True, the scale is fixed to an initial estimate. If False
            (default), the scale is a free fit parameter.
        verbose : bool (optional)
            If True, output additional debugging information.
        guess_length_scale : float (optional)
            The initial length scale to use for the fit. The default is 20
            days.
        preprocessing_kwargs : kwargs (optional)
            Additional preprocessing arguments that are passed to
            `preprocess_observations`.

        Returns
        -------
        gaussian_process : function
            A Gaussian process conditioned on the object's lightcurve. This is
            a wrapper around the george `predict` method with the object flux
            fixed.
        gp_observations : pandas.DataFrame
            The processed observations that the GP was fit to. This could have
            effects such as background subtraction applied to it.
        gp_fit_parameters : list
            A list of the resulting GP fit parameters.
        """
        gp_observations = self.preprocess_observations(**preprocessing_kwargs)

        fluxes = gp_observations["flux"]
        flux_errors = gp_observations["flux_error"]

        wavelengths = gp_observations["band"].map(get_band_central_wavelength)
        times = gp_observations["time"]

        # Use the highest signal-to-noise observation to estimate the scale. We
        # include an error floor so that in the case of very high
        # signal-to-noise observations we pick the maximum flux value.
        signal_to_noises = np.abs(fluxes) / np.sqrt(
            flux_errors ** 2 + (1e-2 * np.max(fluxes)) ** 2
        )
        scale = np.abs(fluxes[signal_to_noises.idxmax()])

        kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
            [guess_length_scale ** 2, 6000 ** 2], ndim=2
        )

        if fix_scale:
            kernel.freeze_parameter("k1:log_constant")
        kernel.freeze_parameter("k2:metric:log_M_1_1")

        gp = george.GP(kernel)

        guess_parameters = gp.get_parameter_vector()

        if verbose:
            print(kernel.get_parameter_dict())

        x_data = np.vstack([times, wavelengths]).T

        gp.compute(x_data, flux_errors)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        bounds = [(0, np.log(1000 ** 2))]
        if not fix_scale:
            bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds

        fit_result = minimize(
            neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds
        )

        if fit_result.success:
            gp.set_parameter_vector(fit_result.x)
        else:
            # Fit failed. Print out a warning, and use the initial guesses for
            # fit parameters. This only really seems to happen for objects
            # where the lightcurve is almost entirely noise.
            logger.warn(
                "GP fit failed for %s! Using guessed GP parameters. "
                "This is usually OK." % self
            )
            gp.set_parameter_vector(guess_parameters)

        if verbose:
            print(fit_result)
            print(kernel.get_parameter_dict())

        # Return the Gaussian process and associated data.
        gaussian_process = partial(gp.predict, fluxes)

        return gaussian_process, gp_observations, fit_result.x

    def get_default_gaussian_process(self):
        """Get the default Gaussian Process.

        This method calls fit_gaussian_process with the default arguments and
        caches its output so that multiple calls only require fitting the GP a
        single time.
        """
        if self._default_gaussian_process is None:
            gaussian_process, _, _ = self.fit_gaussian_process()
            self._default_gaussian_process = gaussian_process

        return self._default_gaussian_process

    def predict_gaussian_process(
        self, bands, times, uncertainties=True, fitted_gp=None, **gp_kwargs
    ):
        """Predict the Gaussian process in a given set of bands and at a given
        set of times.

        Parameters
        ==========
        bands : list(str)
            bands to predict the Gaussian process in.
        times : list or numpy.array of floats
            times to evaluate the Gaussian process at.
        uncertainties : bool (optional)
            If True (default), the GP uncertainties are computed and returned
            along with the mean prediction. If False, only the mean prediction
            is returned.
        fitted_gp : function (optional)
            By default, this function will perform the GP fit before doing
            predictions. If the GP fit has already been done, then the fitted
            GP function (returned by fit_gaussian_process) can be passed here
            instead to skip redoing the fit.
        gp_kwargs : kwargs (optional)
            Additional arguments that are passed to `fit_gaussian_process`.

        Returns
        =======
        predictions : numpy.array
            A 2-dimensional array with shape (len(bands), len(times))
            containing the Gaussian process mean flux predictions.
        prediction_uncertainties : numpy.array
            Only returned if uncertainties is True. This is an array with the
            same shape as predictions containing the Gaussian process
            uncertainty for the predictions.
        """
        if fitted_gp is not None:
            gp = fitted_gp
        else:
            gp, _, _ = self.fit_gaussian_process(**gp_kwargs)

        # Predict the Gaussian process band-by-band.
        predictions = []
        prediction_uncertainties = []

        for band in bands:
            wavelengths = np.ones(len(times)) * get_band_central_wavelength(band)
            pred_x_data = np.vstack([times, wavelengths]).T
            if uncertainties:
                band_pred, band_pred_var = gp(pred_x_data, return_var=True)
                prediction_uncertainties.append(np.sqrt(band_pred_var))
            else:
                band_pred = gp(pred_x_data, return_cov=False)
            predictions.append(band_pred)

        predictions = np.array(predictions)
        if uncertainties:
            prediction_uncertainties = np.array(prediction_uncertainties)
            return predictions, prediction_uncertainties
        else:
            return predictions