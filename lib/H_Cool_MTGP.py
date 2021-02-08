'''
The code in this file uses parts of the sklearn GPR library as a starting point.
'''

import warnings
import sys
import numpy as np
from operator import itemgetter

from sklearn.utils import check_random_state
from scipy.linalg import cholesky, cho_solve, solve_triangular, inv, pinv
from sklearn.base import clone
from sklearn.utils.validation import check_X_y

from lib.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from scipy.optimize import fmin_l_bfgs_b
from copy import deepcopy


"""Gaussian processes regression."""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated
from copy import deepcopy

class GaussianProcessRegressor(BaseEstimator, RegressorMixin):
    """Gaussian process regression (GPR).

    The implementation is based on Algorithm 2.1 of Gaussian Processes
    for Machine Learning (GPML) by Rasmussen and Williams.

    In addition to standard scikit-learn estimator API,
    GaussianProcessRegressor:

       * allows prediction without prior fitting (based on the GP prior)
       * provides an additional method sample_y(X), which evaluates samples
         drawn from the GPR (prior or posterior) at given inputs
       * exposes a method log_marginal_likelihood(theta), which can be used
         externally for other ways of selecting hyperparameters, e.g., via
         Markov chain Monte Carlo.

    Read more in the :ref:`User Guide <gaussian_process>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Targcopet values in training data (also required for prediction)

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``

    """
    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):
        self.kernel = kernel
        self.alpha = alpha
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.random_state = random_state

    
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_
            kernel = self.kernelX_

        else:
            kernel = self.kernelX_.clone_with_theta(theta)

        # Evaluate gradiente with original X
        if eval_gradient:
            K, K_gradient = kernel(self.XX, eval_gradient=True)
        else:
            K = kernel(self.XX, self.XX)
        
        K[np.diag_indices_from(K)] += self.alpha
        
        # Support multi-dimensional output of self.y_train_
        y_train = deepcopy(self.y_train_)
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        # Remove the mean on y
        N,D = self.XX.shape
        y_ant = (self.alpha_ @ self.X_train_[:,D:])@(self.X_train_[:,D:].T)
        y_train -= y_ant[:, np.newaxis]

        #L = self.L_
        #L_inv = solve_triangular(self.L_.T, np.eye(self.L_.shape[0]))
        #K_inv = L_inv.dot(L_inv.T)
        #K_det = np.log(np.diag(L)).sum()
        K_inv = self.K_inv2 
        K_det = self.K_det2
        
        alpha = self.alpha_[:,np.newaxis]
       
        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)

        log_likelihood_dims -= K_det
        log_likelihood_dims -= K.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
           
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)  # k: output-dimension

            tmp -= K_inv[:, :, np.newaxis]
            #tmp -= cho_solve((L, True), np.eye(K.shape[0]))[:, :, np.newaxis]
            
            # Compute "0.5 * trace(tmp.dot(K_gradient))" without
            # constructing the full matrix tmp.dot(K_gradient) since only
            # its diagonal is required
            log_likelihood_gradient_dims = \
                0.5 * np.einsum("ijl,ijk->kl", tmp, K_gradient)
            log_likelihood_gradient = log_likelihood_gradient_dims.sum(-1)

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood
    
    def getNoiseLevel(self):

        params= self.kernel_.get_params()
        list_keys=params.keys()
        key_noise=[s for s in list_keys if s.endswith('noise_level')]
        if key_noise is None:
            print('The kernel structure is not well defined, the noise level cannot be found')
            exit()

        sigma2=params[key_noise[0]]
        return sigma2

    def getSignalLevel(self):

        params= self.kernel_.get_params()
        list_keys=params.keys()
        key_signal=[s for s in list_keys if s.endswith('constant_value')]
        if key_signal is None:
            print('The kernel structure is not well defined, the signal level cannot be found')
            exit()

        slevel=params[key_signal[0]]
        
        return slevel


################ COOL MTGP ############################

class MultitaskGP:

    """MultiTaskGaussian process regression (MTGPR).

    The implementation is based on the paper "Yet another multitask GP"

    It extends the GP scikit-learn estimator API,
    GaussianProcessRegressor (version:: 0.18)

    Parameters
    ----------
    kernel : kernel object
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.

    alpha : float or array-like, optional (default: 1e-10)
        Value added to the diagonal of the kernel matrix during fitting.
        Larger values correspond to increased noise level in the observations.
        This can also prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        If an array is passed, it must have the same number of entries as the
        data used for fitting and is used as datapoint-dependent noise level.
        Note that this is equivalent to adding a WhiteKernel with c=alpha.
        Allowing to specify the noise level directly as a parameter is mainly
        for convenience and for consistency with Ridge.

    optimizer : string or callable, optional (default: "fmin_l_bfgs_b")
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::

            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func' is the objective function to be maximized, which
                #   takes the hyperparameters theta as parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min

        Per default, the 'fmin_l_bfgs_b' algorithm from scipy.optimize
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are::

            'fmin_l_bfgs_b'

    n_restarts_optimizer : int, optional (default: 0)
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer == 0 implies that one
        run is performed.

    normalize_y : boolean, optional (default: False)
        Whether the target values y are normalized, i.e., the mean of the
        observed target values become zero. This parameter should be set to
        True if the target values' mean is expected to differ considerable from
        zero. When enabled, the normalization effectively modifies the GP's
        prior based on the data, which contradicts the likelihood principle;
        normalization is thus disabled per default.

    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.

    random_state : int, RandomState instance or None, optional (default: None)
        The generator used to initialize the centers. If int, random_state is
        the seed used by the random number generator; If RandomState instance,
        random_state is the random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Attributes
    ----------
    X_train_ : array-like, shape = (n_samples, n_features)
        Feature values in training data (also required for prediction)

    y_train_ : array-like, shape = (n_samples, [n_output_dims])
        Targcopet values in training data (also required for prediction)

    kernel_ : kernel object
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters

    L_ : array-like, shape = (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in ``X_train_``

    alpha_ : array-like, shape = (n_samples,)
        Dual coefficients of training data points in kernel space

    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``


    There are more.... TO BE INCLUDED

    """

    __X =None
    __y = None
    __D = None
    __T = None
    __n_restarts_optimizer = 0
    __optimizer = None
    __listGPs = []
    
    

    def __init__(self, kernel=None, kernel_noise = None, alpha=1e-6,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None):

    #def __init__(self, X, y, kernelX= None, kernelY= None, n_restarts_optimizer=0, optimizer="fmin_l_bfgs_b", 
    #    normalize_y=False, copy_X_train=True, random_state = None, ARDTasks = False):
        '''Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        '''
      
        
        self.__n_restarts_optimizer = n_restarts_optimizer
        self.__optimizer = optimizer
    
        self.__listGPs = []
      
        self.random_state = random_state
        self.normalize_y = normalize_y
        self.copy_X_train = copy_X_train
        self.alpha = alpha

        if kernel is None:  # Use an RBF kernel as default
            kernel = RBF(1.0, length_scale_bounds="fixed") 
       
        if kernel_noise is None:  # Use white noise as default
            kernel_noise = WhiteKernel(1, constant_value_bounds="fixed")
        
        self.kernelX_ = deepcopy(kernel)
        self.kernelNoise_ = deepcopy(kernel_noise)

        if not sys.warnoptions:
            warnings.simplefilter("ignore")
                
                
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        
        if self.__optimizer == "fmin_l_bfgs_b":
            
            theta_opt, func_min, convergence_dict = \
                fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
                 #fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds)
            if convergence_dict["warnflag"] != 0:
                warnings.warn("fmin_l_bfgs_b terminated abnormally with the "
                              " state: %s" % convergence_dict)
        elif callable(self.__optimizer):
            
            theta_opt, func_min = \
                self.__optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.__optimizer)

        return theta_opt, func_min
    
    def setThetas(self, theta_opt):
        self.theta = theta_opt
        
        for t,gp in enumerate(self.__listGPs):
            theta_t = theta_opt[self.thetaMask[t]]
            gp.kernel_ = gp.kernel_.clone_with_theta(theta_t)

        return self

    def setAlphas(self, alpha_opt):
        for t,gp in enumerate(self.__listGPs):
            gp.alpha_ = alpha_opt[t]
        return self

    def setOuputs(self, outputs_opt):
        for t,gp in enumerate(self.__listGPs):
            gp.f_ = outputs_opt[t]
        return self
            
    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """Returns log-marginal likelihood of theta for training data.

        Parameters
        ----------
        theta : array-like, shape = (n_kernel_params,) or None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.

        eval_gradient : bool, default: False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.

        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.

        log_likelihood_gradient : array, shape = (n_kernel_params,), optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when eval_gradient is True.
        """
        if theta is None:
            theta = self.theta
            #if eval_gradient:
            #    raise ValueError(
            #        "Gradient can only be evaluated for theta!=None")
            #return self.log_marginal_likelihood_value_

        if not np.array_equal(theta, self.theta):
            # Update alpha for this theta 
            self.computeAlphas(theta = theta, verbose=False)  
        
        thetaMask = self.thetaMask
        log_likelihood=0
        log_likelihood_gradient =np.zeros(theta.shape)

        # Update inverses (if needed) for computation speed up
        
        for t,gp in enumerate(self.__listGPs):
            theta_t = theta[thetaMask[t,:]]
            
            if eval_gradient:
                lml_aux, grad_aux = gp.log_marginal_likelihood(
                    theta_t, eval_gradient=True)
                log_likelihood_gradient[thetaMask[t]] = log_likelihood_gradient[thetaMask[t]] + grad_aux 
            
            else:
                lml_aux = gp.log_marginal_likelihood(
                    theta_t)
            log_likelihood = log_likelihood + lml_aux
            
        
        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood



    def computeAlphas(self, theta = None, verbose=True): 
       
        # Some parameters
        T = len(self.__listGPs)
        D = self.__D
        N = self.__X.shape[0]
        
        # Update kernel parameters
        if theta is not None:            
            for t, gp in enumerate(self.__listGPs):
                theta_t = theta[self.thetaMask[t]]
                gp.kernel_ = gp.kernel_.clone_with_theta(theta_t)
                gp.kernelX_ = gp.kernelX_.clone_with_theta(theta_t)
        

                #print(t)

        if self.alpha_method == 'largeN':
        
            # Compute inverse of K = Kx + noise I for likelihood
            # Get original kernel of X and update theta and SVD (if needed)
            thetaMask_X = np.all(self.thetaMask,axis=0)

            #if np.all(thetaMask_X):
            kernelX = self.kernelX_.clone_with_theta(theta[thetaMask_X])
            # Compute SVD of this kernel matrix
            gp = self.__listGPs[0]
            Kx = kernelX(gp.X_train_)
            Kx[np.diag_indices_from(Kx)] += gp.alpha
            u_K, s_K, _ = np.linalg.svd(Kx, hermitian=True)
            #else:
            #    u_K = self.__u_K
            #    s_K = self.__s_K


            for t, gp in enumerate(self.__listGPs):
                # Get kernel of first GP and compute SVD
                # The remaining SVDs are computed with a rank one update

                NoiseValue = gp.getNoiseLevel() 
                AValue = gp.getSignalLevel()
                                   
                diag = AValue*s_K+NoiseValue + gp.alpha
                K_det_aux = 0.5*np.log(diag).sum()

                K_inv_aux = u_K*np.divide(1,diag)@u_K.T
                gp.K_inv2 = K_inv_aux 
                gp.K_det2 = K_det_aux 
                gp.thetaMask_X = thetaMask_X[self.thetaMask[t]]
                
                Y = self.__y   
               
                if t==0:
                    gp.K_inv = gp.K_inv2 
                    gp.K_det = gp.K_det2           
                else:
                    K_inv_aux_Y = gp.K_inv2 @ Y[:,:t]
                    K_tt = np.eye(t) +Y[:,:t].T @ K_inv_aux_Y
                    K_inv = gp.K_inv2 - K_inv_aux_Y @ np.linalg.inv( K_tt) @ K_inv_aux_Y.T
                    K_det =  gp.K_det2 + 0.5*np.log(np.linalg.det(K_tt))
                    gp.K_inv = K_inv
                    gp.K_det = K_det

                gp.alpha_ = gp.K_inv @ gp.y_train_
                y_ant = (gp.alpha_ @ gp.X_train_[:,D:])@(gp.X_train_[:,D:].T)
                gp.alphax_ = gp.K_inv2 @ (gp.y_train_-y_ant)
                

        else:
            
            for t, gp in enumerate(self.__listGPs):
                Kt = gp.kernel_(gp.X_train_)
                Kt[np.diag_indices_from(Kt)] += gp.alpha
            
                try:
                    L_ = cholesky(Kt, lower=True)  # Line 2
                    L_0 = L_.copy()
                    L_inv = solve_triangular(L_.T, np.eye(L_.shape[0]))
                    #gp.K_inv = L_inv.dot(L_inv.T)
                    #gp.K_det = np.log(np.diag(L_)).sum()
                    gp.alpha_ = cho_solve((L_, True), gp.y_train_.copy())  # Line 3
                    #gp.alpha_ = gp.K_inv @ gp.y_train_
                except np.linalg.LinAlgError as exc:
                    '''exc.args = ("The kernel, %s, is not returning a "
                                "positive definite matrix. Try gradually "
                                "increasing the 'alpha' parameter of your "
                                "GaussianProcessRegressor estimator."
                                % gp.kernel_,) + exc.args'''
                    gp.K_inv = np.linalg.pinv(Kt)
                    #gp.K_det = 0.5*np.log(np.linalg.det(Kt))
                    #gp.alpha_ = cho_solve((L_, True), gp.y_train_.copy())  # Line 3
                    gp.alpha_ = gp.K_inv @ gp.y_train_
                    #raise
                # Save some variables    
                #gp.L_ = L_.copy()

                Kt2 = gp.kernelX_(gp.XX)
                Kt2[np.diag_indices_from(Kt2)] += gp.alpha

                try:
                    L2_ = cholesky(Kt2, lower=True)  # Line 2
                    L2_0 = L2_.copy()
                    L2_inv = solve_triangular(L2_.T, np.eye(L2_.shape[0]))
                    gp.K_inv2 = L2_inv.dot(L2_inv.T)
                    gp.K_det2 = np.log(np.diag(L2_)).sum()
                    #gp.alpha_ = cho_solve((L_, True), gp.y_train_.copy())  # Line 3
                    
                except np.linalg.LinAlgError as exc:
                    '''exc.args = ("The kernel, %s, is not returning a "
                                "positive definite matrix. Try gradually "
                                "increasing the 'alpha' parameter of your "
                                "GaussianProcessRegressor estimator."
                                % gp.kernel_,) + exc.args'''
                    gp.K_inv2 = np.linalg.pinv(Kt)
                    gp.K_det2 = 0.5*np.log(np.linalg.det(Kt))
                    #gp.alpha_ = cho_solve((L_, True), gp.y_train_.copy())  # Line 3
                    

                y_ant = (gp.alpha_ @ gp.X_train_[:,D:])@(gp.X_train_[:,D:].T)
                gp.alphax_ = gp.K_inv2 @ (gp.y_train_-y_ant)
                
                
    '''
    def gramsmith(self, V):
        # Returns a orthogonalized version of V, so that U.T @ U = I
        V[:,0] = V[:,0]/np.linalg.norm(V[:,0])
        for i in range(1,V.shape[0]):
            V[:,i]=V[:,i]- V[:,:i]@V[:,:i].T@V[:,i]
            V[:,i] = V[:,i]/np.linalg.norm(V[:,i])

        return V
    
    ''' 

    def fit (self, X, y,  alpha_method= None):
        """Fit Gaussian process regression model.  
        Returns
        -------
        self : returns an instance of self.
        """

        self.__D = X.shape[1]
        self.__T = y.shape[1] 

        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
              
       
        
        self.__X = np.copy(X) if self.copy_X_train else X
        self.__y = np.copy(y) if self.copy_X_train else y

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            self.__y = self.__y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(self.__T)
        
        if np.iterable(self.alpha) and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))
        
        
        # Prefix method to compute alpha
        if alpha_method == 'largeN':
            self.alpha_method = 'largeN'
        if alpha_method == 'largeT':
            self.alpha_method = 'largeT'
        else:
            N, T = self.__y.shape
            if (T*(N**3)>0.25*(T**4)):
                self.alpha_method = 'largeN'
            else:
                self.alpha_method = 'largeT'


        self.__listGPs =[]

        
        # Define combined kernels for the differnet tasks using masks
        kernelY_ = DotProduct(sigma_0=0, sigma_0_bounds="fixed")
        kernelSignal = ConstantKernel(1, (0.000001, 1000))  # Use to model intratask signal matrix
        kernelX_ = deepcopy(self.kernelX_)
        kernel_noise_ = deepcopy(self.kernelNoise_)

        # Default kernel 
        maskX = np.ones((self.__D,),dtype=bool)
        self.kernelX_.setMask(maskX)
        kernel_0 = deepcopy(self.kernelX_*kernelSignal + self.kernelNoise_)

        # For t=0....T
        for t in range(self.__T):

            if t>0:
                # Define joint kernel 
                # These mask indicate which variables of Z = [X y_:t] will be used for each kernel
                maskX = np.hstack((np.ones((self.__D,), dtype=bool), np.zeros((t,), dtype=bool)))
                maskY = np.hstack((np.zeros((self.__D,), dtype=bool), np.ones((t,), dtype=bool)))
                
                kernelX_.setMask(maskX)
                kernelY_.setMask(maskY)

                # Noise kernel does not need mask
                
                kernel_t = deepcopy(kernelX_*kernelSignal + self.kernelNoise_ + kernelY_)
                self.theta = np.append(np.append(self.theta, kernelSignal.theta), self.kernelNoise_.theta)
                self.bounds = np.concatenate((self.bounds, kernelSignal.bounds, self.kernelNoise_.bounds),axis=0)
                # To update the mask, we firstly set to zero the new theta of the previous tasks
                self.thetaMask = np.concatenate((self.thetaMask, np.zeros((t,self.theta.shape[0]-self.thetaMask.shape[1]))),axis=1)
                # Then include new row for new task
                thetaMaskNew = np.zeros((1,self.thetaMask.shape[1]))
                thetaMaskNew[0,:kernelX_.theta.shape[0]]=1
                thetaMaskNew[0,-(kernelSignal.theta.shape[0]+self.kernelNoise_.theta.shape[0]):]=1
                self.thetaMask = np.concatenate((self.thetaMask, thetaMaskNew),axis=0)
                
            else:
                
                kernel_t = kernel_0

                # Define common theta, bounds and the mask indicating which parameter is optimized by each gp
                self.theta = deepcopy(kernel_t.theta)
                self.bounds = deepcopy(kernel_t.bounds)
                self.thetaMask = np.ones((1,kernel_t.theta.shape[0]))
                

            gp = GaussianProcessRegressor(kernel=kernel_t, n_restarts_optimizer=self.__n_restarts_optimizer, optimizer=self.__optimizer)
            gp.kernel_ = clone(gp.kernel)
            
            gp.XX = self.__X 
            gp.kernelX_ = kernel_0
            gp.alpha = self.alpha
            self.__listGPs.append(gp)
            # Combine X and Y as input
            Z = np.hstack((X, y[:, :t]))
            gp.X_train_ = np.copy(Z) if self.copy_X_train else Z
            gp.y_train_ = np.copy(y[:, t]) if self.copy_X_train else y[:, t]

            gp._y_train_mean = self._y_train_mean[t]

        self.thetaMask = self.thetaMask.astype(bool)
        thetaMask_X = np.all(self.thetaMask,axis=0)

        # If kernel_x has no parameters, precompute here the SVD of Kx for the inference process
        if ~np.all(thetaMask_X) and (self.alpha_method == 'largeN'):
            print ('Precoumpute input kernel SVD')
            kernelX = self.kernelX_
            # Compute SVD of this kernel matrix
            gp = self.__listGPs[0]
            Kx = kernelX(gp.X_train_)
            Kx[np.diag_indices_from(Kx)] += gp.alpha
            u_K, s_K, _ = np.linalg.svd(Kx, hermitian=True)
            #import ipdb; ipdb.set_trace()
            u_K = self.gramsmith(u_K)
            self.__u_K = u_K 
            self.__s_K = s_K



        if self.__optimizer is not None:
        #if self.__optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            
            def obj_func(theta, eval_gradient=True):         
                
                # Update alpha for this theta 
                self.computeAlphas(theta = theta, verbose=False) 

                # 
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(theta = theta, eval_gradient= eval_gradient)
                    return -lml, -grad
                else:
                    lml = self.log_marginal_likelihood(theta = theta, eval_gradient= eval_gradient)
                    return -lml
                
   
                
            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.theta,
                                                      self.bounds))]
           
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.__n_restarts_optimizer > 0:
                if not np.isfinite(self.kernelX_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.bounds
                for iteration in range(self.__n_restarts_optimizer):
                    
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                       self._constrained_optimization(obj_func,
                                                      theta_initial,
                                                      bounds))
                    
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))

            self.setThetas(optima[np.argmin(lml_values)][0])   
            
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.theta)
        
        
        # Compute alpha values for the adjusted parameters
        self.computeAlphas(theta = self.theta, verbose=True)
        #for tt, gp in enumerate(self.__listGPs):
        #    print(tt)
        #    print(gp.alpha_)
        
        
        
        self.get_MT_Matrix()
        

        #return self


    def __MT_Matrix(self, K_tr):

        # The covariance matrix has size number data x number tasks x number tasks
        T = len(self.__listGPs)
        N,D = self.__X.shape
        
        SigmaTT = np.zeros((T,T))
        #PosteriorWTT = np.zeros((T,T))
        Mat_wyt = np.zeros((T,T))
        #Mat_wxt = np.zeros((T,D))
        Mat_alpha_prior = np.zeros((T,N))
        Mat_W = np.zeros((T,D))
        
        # Get noise and signal parameters for all gps
        NoiseValues = np.zeros((T,))
        AValues = np.zeros((T,))
        VW  =  np.zeros((T,T))
        

        # Get GPs parameters
        for tt, gp in enumerate(self.__listGPs):
            NoiseValues[tt] = gp.getNoiseLevel() 
            AValues[tt] = gp.getSignalLevel()
            
            wwT = gp.alphax_ @ (K_tr) @ gp.alphax_
            #AValues[tt] = AValues[tt]/np.sqrt(wwT)

            Mat_wyt[tt,:tt] = gp.alpha_ @ gp.X_train_[:,D:] 
            #Mat_wxt[tt,:] = np.sqrt(AValues[tt]) * gp.alpha_ @ gp.X_train_[:,:D]/np.sqrt(wwT)
            Mat_alpha_prior[tt,:] = np.sqrt(AValues[tt]) * gp.alphax_ /np.sqrt(wwT)
        

        B =  np.linalg.inv(np.eye(T)-Mat_wyt) 
        #Mat_W = B @  Mat_wxt
        
        #VW = Mat_W@Mat_W.T

        VW = B @ Mat_alpha_prior @  K_tr @ Mat_alpha_prior.T @ B.T
       
        for tt in range(T):
            if tt==0:
                # First element is the noise of gp 0
                SigmaTT[tt,tt] = NoiseValues[tt] 
                
            else:       
                # Iterative update of the remaining elements
                
                # Compute the value of wyt
                wyt = Mat_wyt[tt,:tt] 
                

                #wyt = gp.alpha_ @ gp.X_train_[:,self.__D:]

                # Compute covariance of t with 1:t-1
                sigma_col = wyt.dot(SigmaTT[:tt,:tt])
                SigmaTT[tt,:tt] = sigma_col 
                SigmaTT[:tt,tt] = sigma_col
                SigmaTT[tt,tt] = NoiseValues[tt] + wyt.dot(SigmaTT[:tt,:tt]).dot(wyt)
                
                
        self._PriorW = VW
        self._SigmaTT = SigmaTT     
        
  


    def get_MT_Matrix(self):

        N = self.__X.shape[0]
        
        
        # Get the basic kernel matrix over the training data and train-test data
        K_basic = deepcopy(self.kernelX_)
        gp = self.__listGPs[0]
   
        numKernelParam = len(K_basic.hyperparameters)
        if (numKernelParam>0) and  (K_basic.hyperparameters[0].name != 'sigma_0'):
            K_basic = K_basic.clone_with_theta(gp.kernel_.theta[:numKernelParam])
        K_tr = K_basic(self.__X)
       
        self.__MT_Matrix(K_tr)
 
        PriorW = self._PriorW
        SigmaTT = self._SigmaTT 
        
        return PriorW, SigmaTT 


    def predict(self, X, conf_intervals=True, compute_C_star=False):
        
        N = self.__X.shape[0]
        n_tst = X.shape[0]
        T = self.__y.shape[1]

        # Get the basic kernel matrix over the training data and train-test data
        K_basic = deepcopy(self.kernelX_)
        gp = self.__listGPs[0]
   
        numKernelParam = len(K_basic.hyperparameters)
        if (numKernelParam>0) and  (K_basic.hyperparameters[0].name != 'sigma_0'):
            K_basic = K_basic.clone_with_theta(gp.kernel_.theta[:numKernelParam])
        K_tr = K_basic(self.__X)
        K_tr += gp.alpha
        K_tr_test = K_basic(self.__X, X)
        K_test = K_basic(X)

        # Get the intratask covariance matrices
        PriorW = self._PriorW
        SigmaTT = self._SigmaTT 
        
        # Compute kron products
        C_K_tr = np.kron(PriorW, K_tr)
        C_K_tr_test = np.kron(PriorW, K_tr_test)
        C_K_test = np.kron(PriorW, K_test)
        Sigma_I = np.kron(SigmaTT, np.eye(N))
        Sigma_I_tst = np.kron(SigmaTT, np.eye(n_tst))

        # Get matrix inverse: ojo, hay que poner pinv por si está mal condicionada la matriz en casos con poco ruido
        Inverse = pinv(C_K_tr + Sigma_I)
        
        # Compute mean and std of the MT-GP
        mpred = C_K_tr_test.T @ Inverse @ self.__y.T.ravel()
        mpred = mpred.reshape((n_tst, T), order='F')

        if compute_C_star:
            C_star = C_K_test - C_K_tr_test.T @ Inverse @ C_K_tr_test
            C_star += Sigma_I_tst
            self.C_star_ = C_star
            self.C_K_test_ = C_K_test

        if conf_intervals:
            cov = np.zeros((n_tst, T))
            for t in range(T):
                C_t = PriorW[t, t]
                Sigma_t = SigmaTT[t, t]
            
                C_K_tr = np.kron(C_t, K_tr)
                C_K_tr_test = np.kron(C_t, K_tr_test)
                C_K_test = np.kron(C_t, K_test)
                Sigma_I = np.kron(Sigma_t, np.eye(N))
                Sigma_I_tst = np.kron(Sigma_t, np.eye(n_tst))
                Inverse = pinv(C_K_tr + Sigma_I)
                
                full_cov = C_K_test - C_K_tr_test.T @ Inverse @ C_K_tr_test + Sigma_I_tst
                cov[:, t] = np.diag(full_cov)
            
            lower = mpred - 2*np.sqrt(cov)
            upper = mpred + 2*np.sqrt(cov)

            return mpred, lower, upper

        else:
            return mpred

    def compute_kernel(self, X_1, X_2):
        # Get the basic kernel matrix over the training data and train-test data
        K_basic = deepcopy(self.kernelX_)
        gp = self.__listGPs[0]
   
        numKernelParam = len(K_basic.hyperparameters)
        if (numKernelParam>0) and  (K_basic.hyperparameters[0].name != 'sigma_0'):
            K_basic = K_basic.clone_with_theta(gp.kernel_.theta[:numKernelParam])

        K_1_2 = K_basic(X_1, X_2)

        return K_1_2
    
    def getGPs(self):
        return self.__listGPs




    