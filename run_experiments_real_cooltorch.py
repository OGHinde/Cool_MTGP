#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Óscar García Hinde (oghinde@tsc.uc3m.es)

Run experiments for the real datasets.
This script trains the following models:
    - Independent GPs.
    - Standard Multitask Gaussian Process.
    - Cool Multitask Gaussian Process.

Input arguments are:
    - argv[1] = dataset name
"""

import os
import sys
import warnings
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Git/Utilities/')
warnings.filterwarnings("ignore")
import pickle
import json
import random
import numpy as np
import scipy as sp
from lib.kernels import DotProduct as DotProductMTGP 
from lib.kernels import WhiteKernel as WhiteKernelMTGP
from lib.kernels import RBF as RBFMTGP
from lib.kernels import ConstantKernel as ConstantKernelMTGP
from lib.COOLMTgpr_old import MultitaskGP
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from lib.gpr_mod import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from misc.utils import mean_absolute_percentage_error as MAPE
from misc.utils import time_disp
from sklearn import preprocessing
import math
import torch
import gpytorch
import copy
import time
home = str(Path.home())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def RMSE(Y_true, Y_pred):
    raw_rmse = np.sqrt(MSE(Y_true, Y_pred, multioutput='raw_values'))
    return np.mean(raw_rmse)

def RMSE_NORM(Y_true, Y_pred):
    Y_mean = Y_tst.mean(axis=0)
    raw_rmse = np.sqrt(MSE(Y_true, Y_pred, multioutput='raw_values'))
    return np.mean(raw_rmse/Y_mean)



### STD-MTGP STUFF ###############################################################
##################################################################################

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, X_tr, Y_tr, likelihood, num_tasks, kernel='Linear', rank=1):
        super(MultitaskGPModel, self).__init__(X_tr, Y_tr, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        if kernel=='Linear':
            kernel = gpytorch.kernels.LinearKernel()
        elif kernel=='RBF':
            kernel = gpytorch.kernels.RBFKernel()
        else:
            print('Unkown kernel type.')
            sys.exit()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=num_tasks, rank=rank)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



### COOLTORCH STUFF ##############################################################
##################################################################################

# Define the wrapper to use the factorized MTGP over GPyTorch 
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_t_1, kernel='Linear'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if kernel=='Linear':
            variance_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
            self.kernelX_module = gpytorch.kernels.LinearKernel(variance_prior =variance_prior, active_dims=range(train_x.shape[1]-num_t_1))
            self.kernelX_module.variance = variance_prior.sample()
        elif kernel=='RBF':
            RBF_base = gpytorch.kernels.RBFKernel(active_dims=range(train_x.shape[1]-num_t_1))
            self.kernelX_module = gpytorch.kernels.ScaleKernel(RBF_base)
            # Kernel parameters are initialized later (before optimization) for all the task to be common

            outputscale_prior = gpytorch.priors.GammaPrior(3.0,2.0)
            self.kernelX_module.outputscale = outputscale_prior.sample()
            
        else:
            print('Unkown kernel type.')
            sys.exit()
            
        
        #v_constrains = gpytorch.constraints.Interval(lower_bound=1-np.finfo(float).eps, upper_bound=1+np.finfo(float).eps)
        active_Y_1 = range(train_x.shape[1]-num_t_1,train_x.shape[1]) 
        self.linear_kernelY_1_module = gpytorch.kernels.LinearKernel(active_dims=active_Y_1)#, variance_constraint=v_constrains)
        self.linear_kernelY_1_module.variance = 1  # This parameter is no learnt
        self.X = train_x
        self.Y = train_y
        self.num_t_1 = num_t_1
        self.kernel = kernel


        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.kernelX_module(x) + self.linear_kernelY_1_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def getAmplitude(self):
        if self.kernel=='Linear':
            return self.kernelX_module.variance
        elif self.kernel=='RBF':
            return self.kernelX_module.outputscale

    def getLengthscale(self):
        if self.kernel == 'Linear':
            print('Option not available for Linear kenrel.')
            return None
        elif self.kernel == 'RBF':
            return self.kernelX_module.base_kernel.lengthscale.detach().numpy().item()
    
    def getNoise(self):
        return self.likelihood.noise
    
    def getK(self):
        x = self.X
        K = (self.kernelX_module(x) + self.linear_kernelY_1_module(x)).evaluate()
        K[np.diag_indices_from(K)] += self.getNoise()
        return K
    
    def getK_X(self, Xtest=None):
        x = self.X
        num_t_1 = self.num_t_1
        if self.kernel=='Linear':
            K_X_module = copy.deepcopy(self.kernelX_module)
            K_X_module.variance = 1 
        elif self.kernel=='RBF':
            K_X_module = copy.deepcopy(self.kernelX_module.base_kernel)

        K_X_module = K_X_module.cuda() if (device.type != 'cpu') else K_X_module       
        if Xtest == None:
            return K_X_module(x).evaluate()
        else:
            return K_X_module(x,Xtest).evaluate()
    
    def getAlphas(self):
        x = self.X
        K = self.getK()
        y = self.Y
        #alphas, _ = torch.lstsq(y, K)
        alphas = torch.mm(torch.inverse(K),y.unsqueeze(1))
        return alphas


# Some additional functions for the Cool-MTGP wrapper

def optimizeMTGP(MT_models, MT_likelihood, verbose = True):
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 1000

    MT_optimizer = []
    MT_mll = []
    Joint_loss = 0
    for t in range(T):

        # Find optimal model hyperparameters
        MT_models[t].train()
        MT_likelihood[t].train()

        # Use the adam optimizer
        #optimizer = torch.optim.Adam([
        #    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        #], lr=0.1)
        #optimizer = torch.optim.SGD([
        
        optimizer_t = torch.optim.Adam([
                {"params": MT_models[t].mean_module.parameters()},
                {"params": MT_models[t].likelihood.parameters()},
                {"params" : MT_models[t].kernelX_module.parameters()}
             ], lr=0.1)

        MT_optimizer.append(optimizer_t)
        # "Loss" for GPs - the marginal log likelihood
        mll_t = gpytorch.mlls.ExactMarginalLogLikelihood(MT_likelihood[t], MT_models[t])
        MT_mll.append(mll_t)

    # Force common kernel parameter initialization for all the tasks:
    if MT_models[t].kernel=='RBF':
        lengthscale_prior = gpytorch.priors.GammaPrior(np.sqrt(MT_models[0].X.shape[1]/2), 1.0)
        init_lengthscale = lengthscale_prior.sample()        
        for t in range(T):
            MT_models[t].kernelX_module.base_kernel.lengthscale  = init_lengthscale
        
                
        
    for i in range(training_iter):
        aux_grad_l=[]
        aux_grad_a=[]
        for t in range(T):
            # Zero gradients from previous iteration
            MT_optimizer[t].zero_grad()

            # Output from model
            output = MT_models[t](MT_models[t].X)
    
            # Calc loss and backprop gradients
            loss = -MT_mll[t](output, MT_models[t].Y)
            loss.backward()

            if MT_models[t].kernel=='RBF':
                # In case RBF kernel, joint learning of kernel width
                if t==0:
                    joint_grad = MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad
                    aux_grad_l.append(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad)
                    aux_grad_a.append(MT_models[t].kernelX_module.raw_outputscale.grad) 
                    # print('Gradientes sigma')
                    # print(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad)
                    # print('Gradientes Ctt')
                    # print(MT_models[t].kernelX_module.raw_outputscale.grad)
                else:
                    joint_grad += MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad
                    aux_grad_l.append(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad) 
                    aux_grad_a.append(MT_models[t].kernelX_module.raw_outputscale.grad) 
                    # print('Gradientes sigma')   
                    # print(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad)
                    # print('Gradientes Ctt')
                    # print(MT_models[t].kernelX_module.raw_outputscale.grad)
        
        # print('Gradientes sigma')   
        # print(aux_grad_l)
        # print('Gradientes Ctt')
        # print(aux_grad_l)
        # import ipdb; ipdb.set_trace()

        for t in range(T):
            if MT_models[t].kernel=='RBF':
                MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad = joint_grad/T
            
            MT_optimizer[t].step()

        
            if verbose:
                if MT_models[t].kernel=='Linear':
                    print('Iter %d/%d - Loss: %.3f   amplitude kernel X: %.3f  amplitude kernel Y: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    MT_models[t].kernelX_module.variance.item(),
                    MT_models[t].linear_kernelY_1_module.variance.item(),
                    MT_models[t].likelihood.noise.item()
                    ))
                   
                elif MT_models[t].kernel=='RBF':
                    print('Iter %d/%d - Loss: %.3f   amplitude kernel X: %.3f kernel width: %.3f amplitude kernel Y: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    MT_models[t].kernelX_module.outputscale.item(),
                    MT_models[t].kernelX_module.base_kernel.lengthscale.item(),
                    MT_models[t].linear_kernelY_1_module.variance.item(),
                    MT_models[t].likelihood.noise.item()
                    ))
                    
                else:
                    print('Unkown kernel type.')
                    sys.exit()  

            # In the last iteration, we compute the joint lossvalue
            if i ==  (training_iter-1):   
                Joint_loss += loss.item()

    return MT_models, Joint_loss
 
                


def MT_Matrix(MT_models):

    # The covariance matrix has size number data x number tasks x number tasks
    T = len(MT_models)
    
    t=0
    K_tr = MT_models[t].getK_X()
    x = MT_models[t].X
    y = MT_models[t].Y.unsqueeze(1)
    
    for tt in range(1,T):
        yt = MT_models[tt].Y.unsqueeze(1)
        y = torch.cat((y, yt),1)
    
    N,D = x.shape
    
    SigmaTT = torch.zeros((T,T))
    Mat_wyt = torch.zeros((T,T))
    Mat_alpha_prior = torch.zeros((T,N))
    Mat_W = torch.zeros((T,D))
    
    # Get noise and signal parameters for all gps
    NoiseValues = torch.zeros((T,))
    AValues = torch.zeros((T,))
    VW  =  torch.zeros((T,T))
    
    if (device.type != 'cpu'):
        SigmaTT = SigmaTT.cuda()
        Mat_wyt = Mat_wyt.cuda()
        Mat_alpha_prior = Mat_alpha_prior.cuda()
        Mat_W = Mat_W.cuda()
        
        # Get noise and signal parameters for all gps
        NoiseValues = NoiseValues.cuda()
        AValues = AValues.cuda()
        VW  = VW.cuda()
    

    # Get GPs parameters
    for tt, gp in enumerate(MT_models):
        NoiseValues[tt] = gp.getNoise()
        AValues[tt] = gp.getAmplitude()
        alpha = gp.getAlphas()
        wwT = torch.mm(torch.mm(torch.transpose(alpha, 0, 1), K_tr), alpha)
        y_tt = y[:,:tt]
        Mat_wyt[tt,:tt] = torch.mm(torch.transpose(alpha, 0, 1), y_tt)
        Mat_alpha_prior[tt,:] = (torch.sqrt(AValues[tt]) * alpha /torch.sqrt(wwT)).squeeze()
    
    I = torch.eye(T)
    if (device.type != 'cpu'):
        I = I.cuda()
    
    B =  torch.inverse(I-Mat_wyt) 
   
    aux = torch.mm(B, Mat_alpha_prior)
    VW = torch.mm(torch.mm(aux,K_tr) , torch.transpose(aux,0,1))
    for tt in range(T):
        if tt==0:
            # First element is the noise of gp 0
            SigmaTT[tt,tt] = NoiseValues[tt] 
            
        else:       
            # Iterative update of the remaining elements
            
            # Compute the value of wyt
            wyt = Mat_wyt[tt,:tt].unsqueeze(0)
           
            # Compute covariance of t with 1:t-1
            sigma_col = torch.mm(wyt,SigmaTT[:tt,:tt])
            SigmaTT[tt,:tt] = sigma_col 
            SigmaTT[:tt,tt] = sigma_col
            SigmaTT[tt,tt] = NoiseValues[tt] + torch.mm(sigma_col, torch.transpose(wyt,0,1))
      
    return  VW, SigmaTT 
    
def predict_efficient(Sigma, C, K_tr, K_tr_test, y_tr):

    N = K_tr.shape[0]
    N_test = K_tr_test.shape[1]

    u_sigma, s_sigma, _ = np.linalg.svd(Sigma)#,  hermitian=True)    
    aux1=(u_sigma*(np.divide(1,np.sqrt(s_sigma))))
    C2 =aux1.T@ C @ aux1

    u_C2, s_C2,_ = np.linalg.svd(C2)#, hermitian=True)    
    u_K, s_K, _ = np.linalg.svd(K_tr)#, hermitian=True)

    s_C2_K = np.kron(s_C2, s_K)

    y_tr2=(y_tr@u_sigma*np.divide(1,np.sqrt(s_sigma)))
    vect_y_tr_hat = np.divide(1,s_C2_K+1)*(u_K.T@y_tr2@u_C2).T.ravel()
    #m_star = np.kron(C,K_tr_test.T) @ ((u_K@vect_y_tr_hat.reshape((T,N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T).T.ravel())
    m_star = (K_tr_test.T @(u_K@vect_y_tr_hat.reshape((T, N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T)@C).T.ravel()
   
    return m_star

########################################################################
########################################################################


# Script parameters
dataset = sys.argv[1]   # Name of the dataset
k_mode = sys.argv[2]    # Kernel mode: 'Linear' or 'RBF'
ratio_tr = 0.8          # Ratio of training vs test samples
normalize = True        # Normalize input data and training targets
denormalize_y = False   # Denorm. predicted targets before computing perf. (False in the paper)
n_iters = 10            # Number of random experiments.
n_init = 1              # Number of random inits. for COOLTorch-MTGP
max_samples = 400       # Limit max number of samples in the database.
verbose = False
debug = True

# Random seed for paper experiments set to 42
random.seed(a=42, version=2)

# Select models to train
train_base = True      # Independent GPs
train_bonilla = True   # Std-MTGP
train_MTGP = True      # Cool-MTGP

print(f'\nRUNNING REAL BENCHMARK FOR {dataset}.\n')

print('Loading data...')
load_dir = '/datasets/real/'
load_file = load_dir + f'{dataset}.pickle'
with open(load_file, 'rb') as f:
    data = pickle.load(f)
print(load_file + ' loaded successfully.')

X = data['X']
Y = data['Y']

if np.isnan(X).any() or np.isnan(Y).any():
    print('Data contains missing values!')

# Eliminate constant features
good_idx = X.std(axis=0) != 0
n_bad = len(good_idx) - sum(good_idx)
print(f'There are {n_bad} constant features.')
X = X[:, good_idx]

N_tot, D = X.shape
if N_tot > max_samples:
    print(f'Large dataset. Limiting to {max_samples} total samples.')
    N_tot = max_samples
n_tr = int(N_tot * ratio_tr)
n_tst = N_tot - n_tr
T = Y.shape[1]
if T==1:
    train_bonilla = False

# Limit STD-MTGP rank (more than 5 doesn't usually work very well)
if T > 5:
    std_rank = 5
else:
    std_rank = T

print('\nTraining samples = {}'.format(n_tr))
print('Test samples = {}'.format(n_tst))
print('Features = {}'.format(D))
print('Tasks = {}'.format(T))

# Define paths to save result files
save_dir = home + f'/results/real/{k_mode}/'
if not os.path.exists(save_dir):
    # If directory doesn't exist, create it
    os.makedirs(save_dir)

if denormalize_y:
    path_base = save_dir + '{}_base_denorm.json'.format(dataset)
    path_bonilla = save_dir + '{}_stdmtgp_denorm.json'.format(dataset, k_mode, std_rank)
    path_MTGP = save_dir + '{}_cooltorch_denorm.json'.format(dataset)
else:
    path_base = save_dir + '{}_base.json'.format(dataset)
    path_bonilla = save_dir + '{}_stdmtgp.json'.format(dataset, k_mode, std_rank)
    path_MTGP = save_dir + '{}_cooltorch.json'.format(dataset)

if not debug:
    # Check if experiments have already been performed
    exists_base = os.path.exists(path_base)
    exists_bonilla = os.path.exists(path_bonilla)
    exists_MTGP = os.path.exists(path_MTGP)
else:
    # If in debug mode, force experiment execution
    exists_base = False
    exists_bonilla = False
    exists_MTGP = False


if k_mode == 'RBF':
    # Initialise RBF kernel for the standard GP
    kernel_base = WhiteKernel(1, (1e-2, 1)) + RBF(1.0, length_scale_bounds=(1e-2, 1e2))
elif k_mode == 'Linear':
    # Initialise Linear kernel for the standard GP
    kernel_base = WhiteKernel(1, (1e-2, 1)) + DotProduct(sigma_0=0, sigma_0_bounds="fixed")
else:
    print("Wrong kernel specified. Must be either 'Linear' or 'RBF'.")
    sys.exit()

print('\nRunning tests:')
rmse_base = []
rmse_bonilla = []
rmse_MTGP = []
r2_base = []
r2_bonilla = []
r2_MTGP = []

if k_mode == 'RBF':
    indep_lengthscale = np.empty((n_iters, ))
    stdMTGP_lengthscale = np.empty((n_iters, ))
    coolMTGP_lengthscale = np.empty((n_iters, ))

for i in range(n_iters):
    # Perform n_iters random experiments
    print('\n\nIteration {} of {}'.format(i+1, n_iters))
    idx_tot = list(range(N_tot))
    random.shuffle(idx_tot)
    idx_tr = idx_tot[:n_tr]
    idx_tst = idx_tot[n_tr:n_tr+n_tst]

    X_tr = X[idx_tr, :]
    Y_tr = Y[idx_tr, :]
    X_tst = X[idx_tst, :]
    Y_tst = Y[idx_tst, :]

    if normalize:
        scaler_X = preprocessing.StandardScaler()
        scaler_X.fit(X_tr)
        X_tr = scaler_X.transform(X_tr) 
        X_tst = scaler_X.transform(X_tst)
        
        scaler_Y = preprocessing.StandardScaler()
        scaler_Y.fit(Y_tr)
        Y_tr = scaler_Y.transform(Y_tr)
        Y_tst_norm = scaler_Y.transform(Y_tst)
    
    # Convert data to tensors
    X_tr_tens = torch.from_numpy(X_tr).float().to(device)
    Y_tr_tens = torch.from_numpy(Y_tr).float().to(device)
    X_tst_tens = torch.from_numpy(X_tst).float().to(device)

    if train_base:
        # BASE MODEL (INDEPENDENT GAUSSIAN PROCESSES)
        if exists_base:
            pass
        else:
            print('\nTraining base model...')
            Y_pred_base_norm = np.zeros_like(Y_tst)
            for t in range(T):
                gp = GaussianProcessRegressor(kernel=kernel_base, n_restarts_optimizer=0)
                gp.fit(X_tr, Y_tr[:,t])
                Y_pred_base_norm[:, t] = gp.predict(X_tst)
                
                kernel_indep = gp.kernel_
                if k_mode == 'RBF':
                    try:
                        indep_lengthscale[i] += kernel_indep.k2.length_scale
                    except:
                        indep_lengthscale[i] += kernel_indep.k2.k2.length_scale
            if k_mode == 'RBF':
                indep_lengthscale[i] /= T

            # Compute performance
            if denormalize_y:
                # De-standardize predictions
                Y_pred_base = scaler_Y.inverse_transform(Y_pred_base_norm)
                
                rmse = RMSE(Y_tst, Y_pred_base)
                r2 = R2(Y_tst, Y_pred_base)
            else:
                rmse = RMSE(Y_tst_norm, Y_pred_base_norm)
                r2 = R2(Y_tst_norm, Y_pred_base_norm)
            
            rmse_base.append(rmse)
            rmse_norm_base.append(rmse_norm)
            r2_base.append(r2)
            print(f'RMSE = {rmse}')
            print(f'R2 = {r2}')
    
    if train_bonilla:
        # BONILLA MODEL
        if exists_bonilla:
            pass
        else:
            print('\nTraining Std-MTGP model with {} kernel and rank {}...'.format(k_mode, std_rank))
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
            model = MultitaskGPModel(X_tr_tens, Y_tr_tens, 
                                     likelihood, 
                                     num_tasks=T, 
                                     kernel=k_mode, 
                                     rank=std_rank)
            
            smoke_test = ('CI' in os.environ)
            training_iterations = 2 if smoke_test else 50

            # Find optimal model hyperparameters
            model.train()
            likelihood.train()
        
            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
                    ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            for tr_iter in range(training_iterations):
                optimizer.zero_grad()
                output = model(X_tr_tens)
                loss = -mll(output, Y_tr_tens)
                loss.backward()
                if verbose:
                    print('Iter %d/%d - Loss: %.3f' % (tr_iter + 1, training_iterations, loss.item()))
                optimizer.step()

            noise = model.likelihood.noise.item()
            B = model.covar_module.task_covar_module.covar_factor.detach().numpy()
            v = model.covar_module.task_covar_module.raw_var.detach().numpy()
            C_std = (B @ B.T + np.diag(v))
            Sigma_std = (noise*np.eye(T))
            
            if k_mode == 'RBF':
                ls = model.covar_module.data_covar_module.lengthscale.detach().numpy()[0][0]
                stdMTGP_lengthscale[i] = ls

            # Switch to eval mode
            model.eval()
            likelihood.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                predictions = likelihood(model(X_tst_tens))
                y_pred_kron_tens = predictions.mean

            # Compute performance
            Y_pred_kron_norm = y_pred_kron_tens.data.numpy()
            if denormalize_y:
                # De-standardize predictions
                Y_pred_kron = scaler_Y.inverse_transform(Y_pred_kron_norm)

                rmse = RMSE(Y_tst, Y_pred_kron)
                r2 = R2(Y_tst, Y_pred_kron)
            else:
                rmse = RMSE(Y_tst_norm, Y_pred_kron_norm)
                r2 = R2(Y_tst_norm, Y_pred_kron_norm)
            rmse_bonilla.append(rmse)
            r2_bonilla.append(r2)
            print(f'RMSE = {rmse}')
            print(f'R2 = {r2}')
        
    if train_MTGP:
        # CHAIN RULE MULTITASK GP MODEL
        if exists_MTGP:
            pass
        else:
            print('\nTraining Cool-MTGP...')
            start_time = time.time()
            Joint_loss_best = math.inf
            for n in range (n_init):
                # Define model and train T factorized models
                MT_models = []
                MT_likelihood = []
                for t in range(T):
                    # Initialise likelihood function and multitask GP instances:
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    likelihood.noise = gpytorch.priors.GammaPrior(3.0, 6.0).sample()
                    likelihoodt = copy.copy(likelihood)
                    
                    Z_tr = torch.cat((X_tr_tens, Y_tr_tens[:,:t]),1)
                    Y_tr_t = Y_tr_tens[:,t]
                    modelt = ExactGPModel(Z_tr, Y_tr_t, likelihood,  num_t_1=t, kernel = k_mode)
                    if (device.type != 'cpu'):
                        modelt = modelt.cuda()
                        likelihoodt = likelihoodt.cuda()

                    MT_likelihood.append(likelihoodt)
                    MT_models.append(modelt)
                

                MT_models, Joint_loss = optimizeMTGP(MT_models, MT_likelihood, verbose=verbose)
                #print('Best loss ' + str(Joint_loss_best))
                #print('Init loss ' + str(Joint_loss))
                if Joint_loss < Joint_loss_best:
                    # print('update model')
                    Joint_loss_best = Joint_loss
                    MT_models_best = copy.copy(MT_models)
        
        
            PriorW, SigmaTT  = MT_Matrix(MT_models_best)
        
            elapsed_time = time.time() - start_time

            # Switch to eval mode
            K_tr = MT_models[0].getK_X().detach().numpy()
            K_tr_test = MT_models[0].getK_X(X_tst_tens).detach().numpy()
            #K_tr[np.diag_indices_from(K_tr)] += 1e-9        

            if (device.type != 'cpu'):
                SigmaTT = SigmaTT.cpu()
                PriorW = PriorW.cpu()

            Y_pred_MTGP_norm = predict_efficient(SigmaTT.detach().numpy(), PriorW.detach().numpy(), K_tr, K_tr_test, Y_tr)
            Y_pred_MTGP_norm = Y_pred_MTGP_norm.reshape((n_tst, T), order='F')

            if k_mode == 'RBF':
                coolMTGP_lengthscale[i] = MT_models_best[0].getLengthscale()
            
            # Compute performance
            if denormalize_y:
                # De-standardize predictions
                Y_pred_MTGP = scaler_Y.inverse_transform(Y_pred_MTGP_norm)

                rmse = RMSE(Y_tst, Y_pred_MTGP)
                r2 = R2(Y_tst, Y_pred_MTGP)
            else:
                rmse = RMSE(Y_tst_norm, Y_pred_MTGP_norm)
                r2 = R2(Y_tst_norm, Y_pred_MTGP_norm)
            
            rmse_MTGP.append(rmse)
            r2_MTGP.append(r2)
            print(f'RMSE = {rmse}')
            print(f'R2 = {r2}')

print('\nBenchmarks concluded.')
# Result summary
if train_base:
    if exists_base:
        pass
    else:
        print('\nRESULTS INDEPENDENT GPs:')
        if k_mode == 'RBF':
            print(f'   - Independent GPs lenghtscale = {indep_lengthscale}')
            print(f'   - Independent GPs average lenghtscale = {indep_lengthscale.mean()}')
        print('   - Average R2 = {}'.format(np.mean(r2_base)))
        print('   - Standard deviation = {}'.format(np.std(r2_base)))
        print('   - Average RMSE = {}'.format(np.mean(rmse_base)))
        print('   - Standard deviation = {}'.format(np.std(rmse_base)))

        results = {}
        if k_mode == 'RBF':
            results['lengthscale'] = indep_lengthscale.mean()
        results['rmse_stack'] = rmse_base
        results['rmse_mean'] = np.mean(rmse_base)
        results['rmse_std'] = np.std(rmse_base)
        results['r2_stack'] = r2_base
        results['r2_mean'] = np.mean(r2_base)
        results['r2_std'] = np.std(r2_base)
        with open(path_base, 'w') as f:
            json.dump(results, f, sort_keys=False, indent=4)
if train_bonilla:
    if exists_bonilla:
        pass
    else:
        print('\nRESULTS STD-MTGP:')
        if k_mode == 'RBF':
            print(f'   - STD-MTGP lenghtscale = {stdMTGP_lengthscale}')
            print(f'   - STD-MTGP average lenghtscale = {stdMTGP_lengthscale.mean()}')
        print('   - Average R2 = {}'.format(np.mean(r2_bonilla)))
        print('   - Standard deviation = {}'.format(np.std(r2_bonilla)))
        print('   - Average RMSE = {}'.format(np.mean(rmse_bonilla)))
        print('   - Standard deviation = {}'.format(np.std(rmse_bonilla)))

        results = {}
        if k_mode == 'RBF':
            results['lengthscale'] = stdMTGP_lengthscale.mean()
        results['rank'] = std_rank
        results['rmse_stack'] = rmse_bonilla
        results['rmse_mean'] = np.mean(rmse_bonilla)
        results['rmse_std'] = np.std(rmse_bonilla)
        results['r2_stack'] = r2_bonilla
        results['r2_mean'] = np.mean(r2_bonilla)
        results['r2_std'] = np.std(r2_bonilla)
        with open(path_bonilla, 'w') as f:
            json.dump(results, f, sort_keys=False, indent=4)
if train_MTGP:
    if exists_MTGP:
        pass
    else:
        print('\nRESULTS COOL-MTGP:')
        if k_mode == 'RBF':
            print(f'   - Cool-MTGP lenghtscale = {coolMTGP_lengthscale}')
            print(f'   - Cool-MTGP average lenghtscale = {coolMTGP_lengthscale.mean()}')
        print('   - Average R2 = {}'.format(np.mean(r2_MTGP)))
        print('   - Standard deviation = {}'.format(np.std(r2_MTGP)))
        print('   - Average RMSE = {}'.format(np.mean(rmse_MTGP)))
        print('   - Standard deviation = {}'.format(np.std(rmse_MTGP)))

        results = {}
        if k_mode == 'RBF':
            results['lengthscale'] = coolMTGP_lengthscale.mean()
        results['rmse_stack'] = rmse_MTGP
        results['rmse_mean'] = np.mean(rmse_MTGP)
        results['rmse_std'] = np.std(rmse_MTGP)
        results['r2_stack'] = r2_MTGP
        results['r2_mean'] = np.mean(r2_MTGP)
        results['r2_std'] = np.std(r2_MTGP)
        with open(path_MTGP, 'w') as f:
            json.dump(results, f, sort_keys=False, indent=4)

print('Done!\n')