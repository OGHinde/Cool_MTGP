import sys
import numpy as np
import math
import torch
import gpytorch
import copy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")

# Auxiliary functions for the Cool-MTGP wrapper.

def optimizeMTGP(MT_models, MT_likelihood, verbose = True):
    smoke_test = ('CI' in os.environ)
    training_iter = 2 if smoke_test else 100

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
                else:
                    joint_grad += MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad
                    aux_grad_l.append(MT_models[t].kernelX_module.base_kernel.raw_lengthscale.grad) 
                    aux_grad_a.append(MT_models[t].kernelX_module.raw_outputscale.grad) 

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


def fit(X_tr_tens, Y_tr_tens, verbose=False, n_init=1, k_mode='rbf'):
    
    if not torch.is_tensor(X_tr_tens) or not torch.is_tensor(Y_tr_tens):
        print('Training data must be a pytorch tensor.')
        sys.exit()

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
            modelt = ExactGPModel(Z_tr, Y_tr_t, likelihood,  num_t_1=t, kernel=k_mode)
            if (device.type != 'cpu'):
                modelt = modelt.cuda()
                likelihoodt = likelihoodt.cuda()
    
            MT_likelihood.append(likelihoodt)
            MT_models.append(modelt)
    
        MT_models, Joint_loss = optimizeMTGP(MT_models, MT_likelihood, verbose=verbose)
        if Joint_loss < Joint_loss_best:
            #import ipdb; ipdb.set_trace()
            # WARNING: WE'RE GETTING RANDOM NANs
            print('Update model.')
            Joint_loss_best = Joint_loss
            MT_models_best = copy.copy(MT_models)
        
        if verbose:
            print('\nBest loss ' + str(Joint_loss_best))
            print('Init loss ' + str(Joint_loss))

def predict(Sigma, C, K_tr, K_tr_test, y_tr):

    N = K_tr.shape[0]
    N_test = K_tr_test.shape[1]

    u_sigma, s_sigma, _ = np.linalg.svd(Sigma)
    aux1=(u_sigma*(np.divide(1,np.sqrt(s_sigma))))
    C2 =aux1.T@ C @ aux1

    u_C2, s_C2,_ = np.linalg.svd(C2)
    u_K, s_K, _ = np.linalg.svd(K_tr)

    s_C2_K = np.kron(s_C2, s_K)

    y_tr2=(y_tr@u_sigma*np.divide(1,np.sqrt(s_sigma)))
    vect_y_tr_hat = np.divide(1,s_C2_K+1)*(u_K.T@y_tr2@u_C2).T.ravel()
    #m_star = np.kron(C,K_tr_test.T) @ ((u_K@vect_y_tr_hat.reshape((T,N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T).T.ravel())
    m_star = (K_tr_test.T @(u_K@vect_y_tr_hat.reshape((T, N)).T@u_C2.T@(u_sigma*np.divide(1,np.sqrt(s_sigma))).T)@C).T.ravel()
   
    return m_star
