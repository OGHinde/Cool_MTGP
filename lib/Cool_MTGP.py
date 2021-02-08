#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 17:11:54 2020

@author: vanessa y oscar

Computational cost evaluation

Suitable datasets are:
    - scm20d (Too large)
    

Input arguments are:
    - argv[1] = dataset name
    - argv[2] = number training data
"""

import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import sys
#import h5py
import pickle
import json
import random
import numpy as np
import scipy as sp
import time
import math
import copy

from sklearn.metrics import mean_squared_error as MSE

from sklearn import preprocessing
import math
import torch
import gpytorch
import time
import random



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

            outputscale_prior = gpytorch.priors.GammaPrior(3.0, 2.0)
            self.kernelX_module.outputscale = outputscale_prior.sample()
            
        else:
            print('Unkown kernel type.')
            sys.exit()
            
        
        #v_constrains = gpytorch.constraints.Interval(lower_bound=1-np.finfo(float).eps, upper_bound=1+np.finfo(float).eps)
        active_Y_1 = range(train_x.shape[1]-num_t_1, train_x.shape[1]) 
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
    training_iter = 2 if smoke_test else 50

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
        lengthscale_prior = gpytorch.priors.GammaPrior(MT_models[0].X.shape[1], 1.0)
        init_lengthscale = lengthscale_prior.sample()        
        for t in range(T):
            MT_models[t].kernelX_module.base_kernel.lengthscale  = init_lengthscale
                
        
    for i in range(training_iter):
        
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
                else:
                    joint_grad += MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad

        for t in range(T):
            if MT_models[t].kernel=='RBF':
                MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad = joint_grad
            
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

    u_sigma, s_sigma, _ = np.linalg.svd(Sigma)#,  hermitian=True)    
    aux1=(u_sigma*(np.divide(1,np.sqrt(s_sigma))))
    C2 =aux1.T@ C @ aux1

    u_C2, s_C2,_ = np.linalg.svd(C2)#, hermitian=True)    
    u_K, s_K, _ = np.linalg.svd(K_tr)#, hermitian=True)

    s_C2_K = np.kron(s_C2,s_K)

    y_tr2=(y_tr@u_sigma*np.divide(1,np.sqrt(s_sigma)))
    vect_y_tr_hat = np.divide(1,s_C2_K+1)*(u_K.T@y_tr2@u_C2).T.ravel()
    #m_star = np.kron(C,K_tr_test.T) @ ((u_K@vect_y_tr_hat.reshape((T,N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T).T.ravel())
    m_star = (K_tr_test.T @(u_K@vect_y_tr_hat.reshape((T,N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T)@C).T.ravel()
   
    return m_star



###################################################
# Main function
###################################################
if __name__ == "__main__":

    dataset = sys.argv[1]
    N_tr = int(sys.argv[2])
    kernel = sys.argv[3]
    selectDevice = sys.argv[4]


    home = str(Path.home())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if selectDevice == 'cpu':
        device = torch.device("cpu")
    print(device)

    normalize = True
    CoolMTGP_kernel = kernel #'Linear' # Only linear are RBF are supported (so far)
    verbose = False
    n_iters = 10
    n_init = 20  # Number of random initializations
    random.seed(a=0, version=2)
       

    print('\nRUNNING REAL BENCHMARK. Dataset = {}.\n'.format(dataset))

    print('Loading data...')

    load_dir = './'  # Path with the data files
    load_file = '{}.pickle'.format(dataset)

    try:
        with open(load_dir+load_file, 'rb') as f:
            data = pickle.load(f)
        print(load_file + ' loaded successfully.')
    except:
        print('Failed to load dataset!')
        sys.exit()

    X = data['X']
    Y = data['Y']

    if np.isnan(X).any() or np.isnan(Y).any():
        print('Data contains missing values!')

    ntot, D = X.shape
    if ntot <= N_tr:
        print('Reduce N_tr, there are not enough training data')
        sys.exit()
    else:  
        n_tr = N_tr
        n_tst = ntot-N_tr

    T = Y.shape[1]


    print('\nTraining samples = {}'.format(n_tr))
    print('Test samples = {}'.format(n_tst))
    print('Features = {}'.format(D))
    print('Tasks = {}'.format(T))

    # Define paths and variables to save result files
    results_dir = 'results/'
    path = results_dir + '{}_CoolMTGP_{}.json'.format(dataset, N_tr)
    
    mse_CoolMTGP = []
    time_CoolMTGP = []
    seed = time.time()
    torch.manual_seed(seed)

    print('\nRunning tests:')

    for i in range(n_iters):
        print('\n\nIteration {} of {}'.format(i+1, n_iters))
        # Random partitioning of the data.
        idx_tot = list(range(ntot))
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
            Y_tst = scaler_Y.transform(Y_tst)
        

        # Convert data to tensors
        X_tr_tens = torch.from_numpy(X_tr).float().to(device)
        Y_tr_tens = torch.from_numpy(Y_tr).float().to(device)
        X_tst_tens = torch.from_numpy(X_tst).float().to(device)

        if (device.type != 'cpu'):
            X_tr_tens = X_tr_tens.cuda()
            Y_tr_tens = Y_tr_tens.cuda() 
            X_tst_tens = X_tst_tens.cuda()        
        
        print('\nTraining Cool MTGP model with {} kernel...'.format(CoolMTGP_kernel))
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
                modelt = ExactGPModel(Z_tr, Y_tr_t, likelihood,  num_t_1=t, kernel = CoolMTGP_kernel)
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
        N = K_tr.shape[0]
        N_test = K_tr_test.shape[1]

        if (device.type != 'cpu'):
            SigmaTT = SigmaTT.cpu()
            PriorW = PriorW.cpu()

        y_pred = predict_efficient(SigmaTT.detach().numpy(), PriorW.detach().numpy(), K_tr, K_tr_test, Y_tr)
        # Compute performance
        mse = MSE(Y_tst, y_pred.reshape((T,n_tst)).T)
    
        mse_CoolMTGP.append(mse)    
        print('MSE = {}'.format(mse))

        time_CoolMTGP.append(elapsed_time)
        print('Training time = {}'.format(elapsed_time))
        

       
    print('\nBenchmarks concluded.')


    print('\nRESULTS Cool MTGP:')
    print('   - Average MSE = {}'.format(np.mean(mse_CoolMTGP)))
    print('   - Standard deviation = {}'.format(np.std(mse_CoolMTGP)))
    print('   - Average Time = {}'.format(np.mean(time_CoolMTGP)))
    

    results = {}
    results['mse_stack'] = mse_CoolMTGP
    results['mse_mean'] = np.mean(mse_CoolMTGP)
    results['mse_std'] = np.std(mse_CoolMTGP)
    results['time_stack'] = time_CoolMTGP
    results['time_mean'] = np.mean(time_CoolMTGP)
    results['time_std'] = np.std(time_CoolMTGP)
    
    with open(path, 'w') as f:
        json.dump(results, f, sort_keys=False, indent=4)


    print('Done!\n')