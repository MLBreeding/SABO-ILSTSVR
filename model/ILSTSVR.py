import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, RegressorMixin 
import KernelFunction as kf
import Hyperplane1
import Hyperplane2

class ILSTSVR(BaseEstimator, RegressorMixin):
    def __init__(self,Epsilon0 = 0.001, Epsilon1=0.1, Epsilon2=0.1, C1=1, C2=1, C3=1, C4=1,kernel_type=2,kernel_param=1,regulz1=0.0001, regulz2=0.0001, Kmax = 100):
        self.Epsilon0 = Epsilon0
        self.Epsilon1=Epsilon1
        self.Epsilon2=Epsilon2
        self.C1=C1
        self.C2=C2
        self.C3 = C3
        self.C4 = C4
        self.regulz1 = regulz1
        self.regulz2 = regulz2
        self.Kmax = Kmax
        self.kernel_type=kernel_type
        self.kernel_param=kernel_param
        
    def fit(self, X, Y):
        assert (type(self.Epsilon1) in [float,int,np.float64])
        assert (type(self.Epsilon2) in [float,int,np.float64])
        assert (type(self.C1) in [float,int,np.float64])
        assert (type(self.C2) in [float,int,np.float64])
        assert (type(self.regulz1) in [float,int,np.float64])
        assert (type(self.regulz2) in [float,int,np.float64])
        assert (type(self.kernel_param) in [float,int,np.float64])
        assert (self.kernel_type in [0,1,2,3])
        r_x,c=X.shape
        r_y=Y.shape[0]
        assert (r_x==r_y)
        r=r_x
        
        e=np.ones((r,1))
        
        if(self.kernel_type==0): # no need to cal kernel
            H = np.hstack((X,e))
        else:
            H = np.zeros((r,r))
            
            for i in range(r):
                for j in range(r):
                    H[i][j] = kf.kernelfunction(self.kernel_type,X[i],X[j],self.kernel_param)

        [w1,b1] = Hyperplane1.Hyperplane_1(H,Y,self.C1,self.C3,self.Epsilon1, self.Kmax, self.Epsilon0)
        [w2,b2] = Hyperplane2.Hyperplane_2(H,Y,self.C2,self.C4,self.Epsilon2, self.Kmax, self.Epsilon0)
        self.plane1_coeff_ = w1
        self.plane1_offset_ = b1
        self.plane2_coeff_ = w2
        self.plane2_offset_ = b2
        self.data_ = X

        return self

    def predict(self, X):

        if(self.kernel_type==0): # no need to cal kernel
            S = X
        else:
            S = np.zeros((X.shape[0],self.data_.shape[0]))
            for i in range(X.shape[0]):
                for j in range(self.data_.shape[0]):
                    S[i][j] = kf.kernelfunction(self.kernel_type,X[i],self.data_[j].T,self.kernel_param)


        
        y1 = np.dot(S,self.plane1_coeff_)+ ((self.plane1_offset_)*(np.ones((X.shape[0],1))))

        y2 = np.dot(S,self.plane2_coeff_)+ ((self.plane2_offset_)*(np.ones((X.shape[0],1))))


        return (y1+y2)/2    


