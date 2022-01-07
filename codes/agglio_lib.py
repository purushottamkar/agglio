import numpy as np
import time
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV


def sigmoid(z, B=1):
    # B: temperature
    return 1/(1 + np.exp(-B*z))

def getObj(w, y, X, B=1):
    ip = np.dot(X,w.ravel())
    return mean_squared_error(y, sigmoid(ip,B))

def getGrad(w, y, X, B=1):
    s=sigmoid(np.dot(X,w), B)
    n = X.shape[0]
    return (2/n)*np.dot(X.T, (s-y)*(s-s**2)*B)    

def recompute_labels(y, B):
    # recompute labels for a temperature B
    return sigmoid(np.log(y/(1-y)),B)

def getData(mu, sigma, n, d):
    return sigma * np.random.randn(n, d) + mu

def sigmoid_noisy_post(z, B=1, sigma_noise=0.005, mu_noise=0, alpha=1):
    # B: temperature, noise distr is N(mu_noise,sigma_noise), alpha=fraction of noisy data
    # type 1: noise added to labels
    xi=sigma_noise*np.random.randn(len(z),1)+mu_noise
    supp = np.zeros((len(z),1))
    supp[0:int(len(z)*alpha)]=1
    supp = np.random.permutation(supp)
    xi = np.multiply(xi,supp)
    y = sigmoid(z, B) + xi
    eps = 0.00001 # adjustment for inversion
    y = np.maximum(y, np.zeros_like(y)+eps) # default entries <0 to 0
    y = np.minimum(y, np.ones_like(y)-eps)  # default entries >1 to 1
    return y

def sigmoid_noisy_pre(z, B=1, sigma_noise=0.5, mu_noise=0, frac=1):
    # B: temperature, noise distr is N(mu_noise,sigma_noise), frac=fraction of noisy data
    # type 2: noise added to z
    xi=sigma_noise*np.random.randn(len(z),1)+mu_noise
    supp = np.zeros((len(z),1))
    supp[0:int(len(z)*frac)]=1
    supp = np.random.permutation(supp)
    xi = np.multiply(xi,supp)
    return sigmoid(z+xi, B)
    
#-----------------------------------------------------------------------------------------#
models = {}
hparams = {}
w_radius = 10
#--------------------------------------------------------------------------------------------------#    

#AGGLIO-GD
class AG_GD(BaseEstimator):
    def __init__( self, alpha = 0.1, B_init=0.005, B_step=1.05):
        self.alpha = alpha
        self.B_init = B_init
        self.B_step = B_step

    def fit( self, X, y, w_init, w_star, max_iter = 100, minibatch_size=20, temp_cap=1):
        X, y = check_X_y(X, y)
        start_time=time.time()
        n, d = X.shape
        t_start=time.time()
        self.distVals=[]
        self.objVals=[]
        self.clock=[]
        self.w=w_init
        B=self.B_init

        for i in range(max_iter):
            self.objVals.append(getObj(self.w, y, X))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            y_B = recompute_labels(y, B)
            grad = getGrad(self.w, y_B, X, B)
            self.w = self.w -  self.alpha/(B*B)*grad
            # projection
            if np.linalg.norm(self.w) > w_radius:
              self.w = self.w/np.linalg.norm(self.w)*w_radius
            B = min(B * self.B_step, temp_cap)
            self.clock.append(time.time()-t_start)
        return self

    def predict(self, X):
        return sigmoid(np.dot(X,self.w))
    
    def score( self, X, y ):
        return -mean_squared_error(y, self.predict(X))    
 

models['AG_GD']=AG_GD
hparams['AG_GD']={}
hparams['AG_GD']['alpha']=np.linspace(start=1, stop=200, num=5).tolist()
hparams['AG_GD']['B_init']=np.power(10.0, [-1, -2, -3, -4]).tolist()
hparams['AG_GD']['B_step']=np.linspace(start=1.01, stop=1.5, num=5).tolist()


#AGGLIO-SGD

class AG_SGD(BaseEstimator):
    def __init__( self, alpha = 0.1, B_init=0.005, B_step=1.05):
        self.alpha = alpha
        self.B_init = B_init
        self.B_step = B_step
        
    def fit( self, X, y, w_init, w_star, max_iter = 100, minibatch_size=20, temp_cap=0.1):
        X, y = check_X_y(X, y)
        n, d = X.shape
        t_start=time.time()
        indices= np.arange(len(X))
        self.distVals=[]
        self.objVals=[]
        self.clock=[]
        self.w=w_init
        B=self.B_init

        for i in range(max_iter):
            self.objVals.append(getObj(self.w, y, X))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            y_B = recompute_labels(y, B)
            # creating minibatch
            indices=np.random.permutation(indices)
            X_bat=X[indices[:minibatch_size]]
            y_B_bat=y_B[indices[:minibatch_size]]
            grad = getGrad(self.w, y_B_bat, X_bat, B)
            self.w = self.w -  self.alpha/(B*B)*grad
            # projection
            if np.linalg.norm(self.w) > w_radius:
              self.w = self.w/np.linalg.norm(self.w)*w_radius
            B = min(B * self.B_step, temp_cap)
            self.clock.append(time.time()-t_start)
        return self

    def predict(self, X):
        return sigmoid(np.dot(X,self.w))
    
    def score( self, X, y ):
        return -mean_squared_error(y, self.predict(X))


models['AG_SGD']=AG_SGD
hparams['AG_SGD']={}
hparams['AG_SGD']['alpha']=np.linspace(start=1, stop=500, num=10).tolist()
hparams['AG_SGD']['B_init']=np.power(10.0, [-1, -2, -3, -4]).tolist()
hparams['AG_SGD']['B_step']=np.linspace(start=1.01, stop=2, num=5).tolist()


# AGGLIO-SVRG
class AG_SVRG(BaseEstimator):
  def __init__( self, alpha = 0.1, B_init=0.005, B_step=1.05, temp_cap=1):
    self.alpha = alpha
    self.B_init = B_init
    self.B_step = B_step
    self.temp_cap= temp_cap


  def fit( self, X, y, w_init, w_star, max_iter = 5, minibatch_size=20, epoch_len=10):
    X, y = check_X_y(X, y)
    n, d = X.shape
    t_start=time.time()
    indices= np.arange(len(X))
    self.distVals=[]
    self.objVals=[]
    self.clock=[]
    self.w=w_init
    B=self.B_init
    w_tau=w_tau_next=w_init

    for i in range(max_iter):            
        y_B = recompute_labels(y, B)                
        grad_mu = getGrad(self.w, y_B, X, B)
        w_tau = w_tau_next
        indx_tau = np.random.randint(0,epoch_len) #other-wise set to epoch_len-1
        for j in range(epoch_len):
          y_B = recompute_labels(y, B)
          self.objVals.append(getObj(self.w, y, X))
          self.distVals.append(np.linalg.norm(self.w - w_star))
          # creating minibatch
          indices=np.random.permutation(indices)
          X_bat=X[indices[:minibatch_size]]
          y_B_bat=y_B[indices[:minibatch_size]]
          grad = getGrad(self.w, y_B_bat, X_bat, B)
          grad_tau = getGrad(w_tau, y_B_bat, X_bat, B)
          self.w = self.w - self.alpha/(B*B) * (grad - grad_tau + grad_mu)
          # projection
          if np.linalg.norm(self.w) > w_radius:
            self.w = self.w/np.linalg.norm(self.w)*w_radius
          if j==indx_tau:
            w_tau_next = self.w
          B = min(B * self.B_step, self.temp_cap)
          self.clock.append(time.time()-t_start)
    return self

  def predict(self, X):
      return sigmoid(np.dot(X,self.w))
    
  def score( self, X, y ):
    return -mean_squared_error(y, self.predict(X))


models['AG_SVRG']=AG_SVRG
hparams['AG_SVRG']={}
hparams['AG_SVRG']['alpha']=np.linspace(start=1, stop=400, num=10).tolist()
hparams['AG_SVRG']['B_init']=np.power(10.0, [-1, -2, -3, -4]).tolist()
hparams['AG_SVRG']['B_step']=np.linspace(start=1.01, stop=2, num=5).tolist()
hparams['AG_SVRG']['temp_cap'] = np.linspace(start=0.4, stop=1, num=5).tolist()


#AGGLIO-ADAM
class AG_ADAM(BaseEstimator):
    def __init__( self, alpha = 0.1, B_init=0.1, B_step=1.01,  beta_1=0.9, beta_2=0.7, epsilon=np.power(10.0, -8)):
        self.alpha = alpha
        self.B_init = B_init
        self.B_step = B_step
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        
    def fit( self, X_train, y_train,  w_init, w_star, max_iter = 100, minibatch_size=20, temp_cap=1):
        X_train, y_train = check_X_y(X_train, y_train)
        
        
        n, d = X_train.shape
        t_start=time.time()
        indices= np.arange(n)
        self.distVals=[]
        self.train_mse=[]
        self.clock=[]
        B=self.B_init
        
        m = np.zeros(shape=(d,))
        v  = np.zeros(shape=(d,))
        self.w=w=w_init
        #train_mse = getObj(w, y_train, X_train)
        
        for i in range(max_iter):
            y_B = recompute_labels(y_train, B)
            # creating minibatch
            indices=np.random.permutation(indices)
            X_bat=X_train[indices[:minibatch_size]]
            y_B_bat=y_B[indices[:minibatch_size]]

            grad = getGrad(w, y_B_bat, X_bat, B)
            
            m=self.beta_1*m + (1-self.beta_1)*grad
            v= self.beta_2*v + (1-self.beta_2)*(grad**2)
            m_hat=m*(1/(1 -np.power(self.beta_1, i+1)))
            v_hat = v*(1/ (1 -np.power(self.beta_2, i+1)))
            w = w -  self.alpha*np.divide(m_hat, np.sqrt(v_hat)+self.epsilon)/(B*B)
            # projection
            if np.linalg.norm(w) > w_radius:
                w = w/np.linalg.norm(w)*w_radius
            B = min(B * self.B_step, temp_cap)
            self.w = w
            self.train_mse.append(getObj(w, y_train, X_train))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            self.clock.append(time.time()-t_start)
        return self

    def predict(self, X):
        return sigmoid(np.dot(X,self.w))
    
    def score( self, X, y ):
        return -mean_squared_error(y, self.predict(X))

    
models['AG_ADAM']=AG_ADAM

hparams['AG_ADAM']={}
hparams['AG_ADAM']['alpha']=np.power(10.0, [0, -1, -2, -3]).tolist()
hparams['AG_ADAM']['B_init']=np.power(10.0, [0, -1, -2, -3]).tolist()
hparams['AG_ADAM']['B_step']=np.linspace(start=1.01, stop=3, num=5).tolist()
hparams['AG_ADAM']['beta_1'] = [0.3, 0.5, 0.7, 0.9]
hparams['AG_ADAM']['beta_2'] = [0.3, 0.5, 0.7, 0.9]
#hparams['AG_ADAM']['epsilon'] = np.power(10.0, [-3,  -5, -8]).tolist()



def init_model(model_name,hparam={},cross_validation=True):
    if model_name=='AG_GD':
        if not cross_validation:
            return AG_GD(alpha= 333, B_init=0.001, B_step=1.2575)
        else:
            return AG_GD(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'])        
    elif model_name=='AG_SGD':
        if not cross_validation:
            return AG_SGD(alpha= 333, B_init=0.001, B_step=1.2575)
        else:
            return AG_SGD(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'])    
    elif model_name=='AG_SVRG':
        if not cross_validation:
            return AG_SVRG(alpha= 89.67, B_init=0.01, B_step=1.01, temp_cap=1 )
        else:
            return AG_SVRG(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'], temp_cap=hparam['temp_cap'])
    elif model_name=='AG_ADAM':
        if not cross_validation:
            return AG_ADAM(alpha= 0.1, B_init=0.1, B_step=1.01, beta_1=0.7, beta_2=0.5 )
        else:
            return AG_ADAM(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'], temp_cap=hparam['temp_cap'])


def cross_validate(X,y,params,cross_validation=False):
    model = models[params['algo']]
    hparam = hparams[params['algo']]
    w0 = params['w0']
    wAst = params['wAst']
 
    if cross_validation:
        cv = ShuffleSplit( n_splits = 1, test_size = 0.3, random_state = 42 )
        grid = GridSearchCV( model(), param_grid=hparam, refit = False, cv=cv) #, verbose=3
        grid.fit( X, y.ravel(), w_init=w0.ravel(), w_star=wAst.ravel(), minibatch_size=50)
        best = grid.best_params_
        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        model_train = init_model(params['algo'],hparam=best)
    else:
        model_train = init_model(params['algo'],cross_validation=False)
    
    model_train.fit( X, y.ravel(), w_init = w0.ravel(), w_star = wAst.ravel(), max_iter=800, minibatch_size=50 )   
    objVals = model_train.objVals
    distVals = model_train.distVals
    time = model_train.clock
    return objVals,distVals,time