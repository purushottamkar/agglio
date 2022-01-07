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

def SiLU(z, B=1):
    return z*sigmoid(z,B)

def getObj(w, y, X, B=1):
    # SiLU
    ip = np.dot(X,w.ravel())
    return mean_squared_error(y, SiLU(ip,B))

def getGrad(w, y, X, B=1):
    # SiLU
    si = SiLU(np.dot(X,w), B)
    s = sigmoid(np.dot(X,w), B)
    n = X.shape[0]
    return (2/n)*np.dot(X.T, (si-y)*(s+B*(s-s**2)))

def getData(mu, sigma, n, d):
    return sigma * np.random.randn(n, d) + mu

#-----------------------------------------------------------------------------------------#


models = {}
hparams = {}
w_radius = 10

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
        self.ipAst = np.matmul(X, w_star)

        for i in range(max_iter):
            self.objVals.append(getObj(self.w, y, X))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            y_B = SiLU(self.ipAst, B).ravel()
            grad = getGrad(self.w, y_B, X, B)
            self.w = self.w -  self.alpha/(B)*grad
            # projection
            if np.linalg.norm(self.w) > w_radius:
              self.w = self.w/np.linalg.norm(self.w)*w_radius
            B = min(B * self.B_step, temp_cap)
            self.clock.append(time.time()-t_start)
        return self

    def predict(self, X):
        return SiLU(np.dot(X,self.w))
    
    def score( self, X, y ):
        return -mean_squared_error(y, self.predict(X))    
 

models['AG_GD']=AG_GD
hparams['AG_GD']={}
hparams['AG_GD']['alpha']=np.linspace(start=0.1, stop=2, num=10).tolist()
hparams['AG_GD']['B_init']=np.power(10.0, [-1, -2]).tolist()
hparams['AG_GD']['B_step']=np.linspace(start=1.01, stop=2, num=5).tolist()
#--------------------------------------------------------------------------------------------------#    

#AGGLIO-SGD

class AG_SGD(BaseEstimator):
    def __init__( self, alpha = 0.1, B_init=0.005, B_step=1.05):
        self.alpha = alpha
        self.B_init = B_init
        self.B_step = B_step


    def fit( self, X, y, w_init, w_star, max_iter = 100, minibatch_size=20):
        X, y = check_X_y(X, y)
        n, d = X.shape
        t_start=time.time()
        indices= np.arange(len(X))
        self.distVals=[]
        self.objVals=[]
        self.clock=[]
        self.w=w_init
        B=self.B_init
        self.ipAst = np.matmul(X, w_star)

        for i in range(max_iter):
            self.objVals.append(getObj(self.w, y, X))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            y_B = SiLU(self.ipAst, B).ravel()
            # creating minibatch
            indices=np.random.permutation(indices)
            X_bat=X[indices[:minibatch_size]]
            y_B_bat=y_B[indices[:minibatch_size]]

            grad = getGrad(self.w, y_B_bat, X_bat, B)
            self.w = self.w -  self.alpha/(B)*grad
            # projection
            if np.linalg.norm(self.w) > w_radius:
              self.w = self.w/np.linalg.norm(self.w)*w_radius
            B = min(B * self.B_step, 1)
            self.clock.append(time.time()-t_start)
        return self

    def predict(self, X):
        return SiLU(np.dot(X,self.w))
    
    def score( self, X, y ):
        return -mean_squared_error(y, self.predict(X))


models['AG_SGD']=AG_SGD
hparams['AG_SGD']={}
hparams['AG_SGD']['alpha']=np.linspace(start=0.1, stop=2, num=10).tolist()
hparams['AG_SGD']['B_init']=np.power(10.0, [-1, -2, -3, -4]).tolist()
hparams['AG_SGD']['B_step']=np.linspace(start=1.01, stop=2, num=5).tolist()
#--------------------------------------------------------------------------------------------------#    

# AGGLIO-SVRG
class AG_SVRG(BaseEstimator):
  def __init__( self, alpha = 0.1, B_init=0.005, B_step=1.05):
    self.alpha = alpha
    self.B_init = B_init
    self.B_step = B_step


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
    self.ipAst = np.matmul(X, w_star)

    for i in range(max_iter):
        y_B = SiLU(self.ipAst, B).ravel()
        grad_mu = getGrad(self.w, y_B, X, B)
        w_tau = w_tau_next
        indx_tau = np.random.randint(0,epoch_len) #other-wise set to epoch_len-1
        for j in range(epoch_len):
            # y_B = recompute_labels(y, B)
            y_B = SiLU(self.ipAst, B).ravel()
            self.objVals.append(getObj(self.w, y, X))
            self.distVals.append(np.linalg.norm(self.w - w_star))
            # creating minibatch
            indices=np.random.permutation(indices)
            X_bat=X[indices[:minibatch_size]]
            y_B_bat=y_B[indices[:minibatch_size]]
            grad = getGrad(self.w, y_B_bat, X_bat, B)
            grad_tau = getGrad(w_tau, y_B_bat, X_bat, B)
            #print(grad.shape, grad_tau.shape, grad_mu.shape)
            self.w = self.w - self.alpha/(B*B) * (grad - grad_tau + grad_mu)
            # projection
            if np.linalg.norm(self.w) > w_radius:
                self.w = self.w/np.linalg.norm(self.w)*w_radius
            if j==indx_tau:
                w_tau_next = self.w
            B = min(B * self.B_step, 1)
            self.clock.append(time.time()-t_start)
    return self

  def predict(self, X):
      return SiLU(np.dot(X,self.w))
    
  def score( self, X, y ):
    return -mean_squared_error(y, self.predict(X))


models['AG_SVRG']=AG_SVRG
hparams['AG_SVRG']={}
hparams['AG_SVRG']['alpha']=np.linspace(start=1, stop=400, num=10).tolist()
hparams['AG_SVRG']['B_init']=np.power(10.0, [-1, -2, -3, -4]).tolist()
hparams['AG_SVRG']['B_step']=np.linspace(start=1.01, stop=2, num=5).tolist()
#--------------------------------------------------------------------------------------------------# 


def init_model(model_name,hparam={},cross_validation=True):
    if model_name=='AG_GD':
        if not cross_validation:
            return AG_GD(alpha= 389.1, B_init=0.1, B_step=1.7525 )
        else:
            return AG_GD(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'])
    elif model_name=='AG_SGD':
        if not cross_validation:
            return AG_SGD(alpha= 333, B_init=0.001, B_step=1.2575 )
        else:
            return AG_SGD(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'])
    elif model_name=='AG_SVRG':
        if not cross_validation:
            return AG_SVRG(alpha= 89.67, B_init=0.01, B_step=1.01 )
        else:
            return AG_SVRG(alpha= hparam['alpha'], B_init=hparam['B_init'], B_step=hparam['B_step'])


def cross_validate(X,y,params,cross_validation=False):
    model = models[params['algo']]
    hparam = hparams[params['algo']]
    w0 = params['w0']
    wAst = params['wAst']
    tmpcap = 1
    if cross_validation:
        cv = ShuffleSplit( n_splits = 1, test_size = 0.3, random_state = 42 )
        grid = GridSearchCV( model(), param_grid=hparam, refit = False, cv=cv) #, verbose=3
        if tmpcap < 1:
          grid.fit( X, y.ravel(), w_init=w0.ravel(), w_star=wAst.ravel(), minibatch_size=50, temp_cap = tmpcap)
        else:
          grid.fit( X, y.ravel(), w_init=w0.ravel(), w_star=wAst.ravel(), minibatch_size=50)
        best = grid.best_params_
        print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
        model_train = init_model(params['algo'],hparam=best)
    else:
        model_train = init_model(params['algo'],cross_validation=False)
    if tmpcap<1:
      model_train.fit( X, y.ravel(), w_init = w0.ravel(), w_star = wAst.ravel(), max_iter=800, minibatch_size=50, temp_cap=tmpcap )   
    else:
      model_train.fit( X, y.ravel(), w_init = w0.ravel(), w_star = wAst.ravel(), max_iter=800, minibatch_size=50 )
    objVals = model_train.objVals
    distVals = model_train.distVals
    time = model_train.clock
    return objVals,distVals,time