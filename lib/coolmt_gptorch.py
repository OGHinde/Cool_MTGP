# General imports

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular, inv, pinv
import math
import torch
import gpytorch
import copy
#import warnings
#warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Running on {device}.')


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
    
    def getK_X(self, Xtest=None, tst_only=False):
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
        elif not tst_only:
            return K_X_module(x, Xtest).evaluate()
        else:
            return K_X_module(Xtest).evaluate()
    
    def getAlphas(self):
        x = self.X
        K = self.getK()
        y = self.Y
        #alphas, _ = torch.lstsq(y, K)
        alphas = torch.mm(torch.inverse(K),y.unsqueeze(1))
        return alphas


# Some additional functions for the Cool-MTGP wrapper
class Cool_MTGP():

    def __init__(self, kernel='Linear'):
        self.k_mode = kernel

    def optimizeMTGP(self, MT_models, MT_likelihood, training_iter, verbose):
        MT_optimizer = []
        MT_mll = []
        Joint_loss = 0
        for t in range(self.n_tasks):
    
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
        if MT_models[t].kernel == 'RBF':
            lengthscale_prior = gpytorch.priors.GammaPrior(np.sqrt(MT_models[0].X.shape[1]/2), 1.0)
            init_lengthscale = lengthscale_prior.sample()        
            for t in range(self.n_tasks):
                MT_models[t].kernelX_module.base_kernel.lengthscale  = init_lengthscale
            
        for i in range(training_iter):
            aux_grad_l=[]
            aux_grad_a=[]
            for t in range(self.n_tasks):
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
                    else:
                        joint_grad += MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad
                        aux_grad_l.append(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad) 
                        aux_grad_a.append(MT_models[t].kernelX_module.raw_outputscale.grad) 
    
            for t in range(self.n_tasks):
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

                
    def MT_Matrix(self, MT_models):
        T = self.n_tasks
        t=0
        K_tr = MT_models[t].getK_X()
        x = MT_models[t].X
        y = MT_models[t].Y.unsqueeze(1)
        
        for tt in range(1, T):
            yt = MT_models[tt].Y.unsqueeze(1)
            y = torch.cat((y, yt),1)
        
        N,D = x.shape
        
        Sigma_tt = torch.zeros((T,T))
        Mat_wyt = torch.zeros((T,T))
        Mat_alpha_prior = torch.zeros((T,N))
        Mat_W = torch.zeros((T,D))
        
        # Get noise and signal parameters for all gps
        NoiseValues = torch.zeros((T,))
        AValues = torch.zeros((T,))
        C_tt  =  torch.zeros((T,T))
        
        if (device.type != 'cpu'):
            Sigma_tt = Sigma_tt.cuda()
            Mat_wyt = Mat_wyt.cuda()
            Mat_alpha_prior = Mat_alpha_prior.cuda()
            Mat_W = Mat_W.cuda()
            
            # Get noise and signal parameters for all gps
            NoiseValues = NoiseValues.cuda()
            AValues = AValues.cuda()
            C_tt  = C_tt.cuda()
        
    
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
        C_tt = torch.mm(torch.mm(aux,K_tr) , torch.transpose(aux,0,1))
        for tt in range(T):
            if tt==0:
                # First element is the noise of gp 0
                Sigma_tt[tt,tt] = NoiseValues[tt] 
                
            else:       
                # Iterative update of the remaining elements
                # Compute the value of wyt
                wyt = Mat_wyt[tt,:tt].unsqueeze(0)
               
                # Compute covariance of t with 1:t-1
                sigma_col = torch.mm(wyt,Sigma_tt[:tt,:tt])
                Sigma_tt[tt,:tt] = sigma_col 
                Sigma_tt[:tt,tt] = sigma_col
                Sigma_tt[tt,tt] = NoiseValues[tt] + torch.mm(sigma_col, torch.transpose(wyt,0,1))
          
        return  C_tt, Sigma_tt

    def train(self, X_tr, Y_tr, n_init=1, training_iter=100, verbose=False):   
        
        print('Training model...')
        _, self.n_tasks = Y_tr.shape
        Joint_loss_best = math.inf
        for n in range (n_init):
            print(f'Iteration {n} of {n_init}.')
            # Define model and train T factorized models
            MT_models = []
            MT_likelihood = []
            for t in range(self.n_tasks):
                # Initialise likelihood function and multitask GP instances:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.noise = gpytorch.priors.GammaPrior(3.0, 6.0).sample()
                likelihoodt = copy.copy(likelihood)
                    
                Z_tr = torch.cat((X_tr, Y_tr[:,:t]),1)
                Y_tr_t = Y_tr[:,t]
                modelt = ExactGPModel(Z_tr, Y_tr_t, likelihood, num_t_1=t, kernel=self.k_mode)
                if (device.type != 'cpu'):
                    modelt = modelt.cuda()
                    likelihoodt = likelihoodt.cuda()
    
                MT_likelihood.append(likelihoodt)
                MT_models.append(modelt)
    
            MT_models, Joint_loss = self.optimizeMTGP(MT_models, MT_likelihood, training_iter, verbose)
            if Joint_loss < Joint_loss_best:
                #import ipdb; ipdb.set_trace()
                # WARNING: WE'RE GETTING RANDOM NANs
                print('Updating model...')
                Joint_loss_best = Joint_loss
                MT_models_best = copy.copy(MT_models)
            
            if verbose:
                print('\nBest loss ' + str(Joint_loss_best))
                print('Init loss ' + str(Joint_loss))
    
        # Get covariance matrices
        C_tt_tens, Sigma_tt_tens  = self.MT_Matrix(MT_models_best)
        if (device.type != 'cpu'):
            C_tt_tens = C_tt_tens.cpu()
            Sigma_tt_tens = Sigma_tt_tens.cpu()
        C_tt = C_tt_tens.detach().numpy()
        # Force symmetry in C_tt to save problems down the line
        C_tt_sym = (C_tt + C_tt.T)/2

        self.Y_tr = Y_tr.detach().numpy()
        self.MT_models_ = MT_models_best
        self.Joint_loss_ = Joint_loss_best
        self.C_tt_ = C_tt_sym
        self.Sigma_tt_ = Sigma_tt_tens.detach().numpy()

    def predict(self, X_tst, compute_C_star=False, conf_intervals=False):
        K_tr = self.MT_models_[0].getK_X().detach().numpy()
        K_tr_test = self.MT_models_[0].getK_X(X_tst).detach().numpy()
        K_test = self.MT_models_[0].getK_X(X_tst, tst_only=True).detach().numpy()
        
        N = K_tr.shape[0]
        N_test = K_tr_test.shape[1]

        u_sigma, s_sigma, _ = np.linalg.svd(self.Sigma_tt_)
        aux1 = (u_sigma * (np.divide(1, np.sqrt(s_sigma))))
        C2 = aux1.T @ self.C_tt_ @ aux1
    
        u_C2, s_C2,_ = np.linalg.svd(C2)
        u_K, s_K, _ = np.linalg.svd(K_tr)
    
        s_C2_K = np.kron(s_C2, s_K)
    
        y_tr2 = (self.Y_tr @ u_sigma * np.divide(1, np.sqrt(s_sigma)))
        vect_y_tr_hat = np.divide(1, s_C2_K + 1) * (u_K.T @ y_tr2 @ u_C2).T.ravel()
        #m_star = np.kron(C,K_tr_test.T) @ ((u_K@vect_y_tr_hat.reshape((T,N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T).T.ravel())
        m_star = (K_tr_test.T @ (u_K @ vect_y_tr_hat.reshape((self.n_tasks, N)).T @ u_C2.T @ (u_sigma * np.divide(1, np.sqrt(s_sigma))).T) @ self.C_tt_).T.ravel()
        m_star = np.reshape(m_star, (N_test, self.n_tasks))

        covariance_data = {}
        if conf_intervals:
            cov = np.zeros((N_test, self.n_tasks))
            for t in range(self.n_tasks):
                C_t = self.C_tt_[t, t]
                Sigma_t = self.Sigma_tt_[t, t]
            
                C_K_tr = np.kron(C_t, K_tr)
                C_K_tr_test = np.kron(C_t, K_tr_test)
                C_K_test = np.kron(C_t, K_test)
                Sigma_I = np.kron(Sigma_t, np.eye(N))
                Sigma_I_tst = np.kron(Sigma_t, np.eye(N_test))
                Inverse = pinv(C_K_tr + Sigma_I)
                
                full_cov = C_K_test - C_K_tr_test.T @ Inverse @ C_K_tr_test + Sigma_I_tst
                cov[:, t] = np.diag(full_cov)
            
            lower = m_star - 2*np.sqrt(cov)
            upper = m_star + 2*np.sqrt(cov)

            covariance_data['upper'] = upper
            covariance_data['lower'] = lower

        if compute_C_star:
            C_K_tr = np.kron(self.C_tt_, K_tr)
            C_K_tr_test = np.kron(self.C_tt_, K_tr_test)
            C_K_test = np.kron(self.C_tt_, K_test)
            Sigma_I = np.kron(self.Sigma_tt_, np.eye(N))
            Sigma_I_tst = np.kron(self.Sigma_tt_, np.eye(N_test))
            Inverse = np.linalg.inv(C_K_tr + Sigma_I)
            Inverse_sym = (Inverse + Inverse.T)/2

            C_star = C_K_test - C_K_tr_test.T @ Inverse_sym @ C_K_tr_test
            C_star += Sigma_I_tst

            covariance_data['C_star'] = C_star
        
        if compute_C_star or conf_intervals:
            return m_star, covariance_data
        else:
            return m_star
