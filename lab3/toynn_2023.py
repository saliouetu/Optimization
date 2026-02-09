#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 20:08:15 2022

@author: benoitmerlet
"""

import numpy as np
from numpy import random as nprd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from copy import deepcopy as dcp

############################################################################
"""                      The class ToyPb                                """
############################################################################

class ToyPb:  
    """ 
    class of simple classification problems (subsets of R^dim) 
    The classification problem is of the form "Does x belongs to Set ?" for x 
    in R^dim and where Set is a subset of R^dim. 
      - dim : integer 
      - f: numerical functions of 'dim' variables (a numpy array of length dim)
        Set is the set of points such that  f(x)<0 (we write Set = {f<0})
      - name : string (optional)
      - bounds : list of 2*dim floats. Limits of the region where x should be 
                 taken
      - loss, loss_prime: two numerical functions of 1 variable 
          (represent the cost function of the error and its derivative)
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            f : numerical function : nd.numpy array |-> float
                (optional if 'name' is given)
            name : string (optional if 'f' is given)
            dim=2 : integer
            bounds=(-1,1) : a tuple of 2 or 2*'dim' floats
            loss, loss_prime : numerical functions: float |-> float.
                (optional)
            loss_name: string
                (optional)
        Does
        -------
        Builds an element of the class ToyPb:
            Set is a subset of R^{dim} where by default dim=2. 
            Set is given either by the parameter 'f' (Set = {f<0})
            or by 'name'
            Possible values for 'name' are 'sin', 'affine', 'square', 'disc', 
            'ring', 'ncube' or 'nball' (these two latter are for classi-
                                        fications pbs in dimension dim>2)
            
            If len(bounds)==2 then self.bounds=(bound[0],bounds[1])*dim.
            
            Possible values for 'loss_name' : 'softplus', 'demanding'.
                        Default is loss(s)=sqrt(s^2 + 1/10) - 1.
        """
        name = kwargs.get('name', None)
        self.name = name
        ######  criteria function  ###################
        dim = kwargs.get("dim",None) or 2
        self.dim =dim 

        f =kwargs.get('f', None)
        if f:
            self.f = f
        elif name == "sin":
            self.f = lambda X : X[1] - .75*np.sin(np.pi*X[0])
        elif name == "affine":
            self.f =lambda X : 2*X[0] - X[1] - .5
        elif name == "disk":
            self.f = lambda X : X[0]**2 + X[1]**2 - .25
        elif name == "square":
            self.f = lambda X : max(np.abs(X[0]), np.abs(X[1])) - .5
        elif name == "ring":
            self.f = lambda X : 9/64 - (13/16)*(X[0]**2 
                                    + X[1]**2) + (X[0]**2 + X[1]**2)**2
        elif name == "ncube":
            self.f = lambda X : max(np.abs(x) for x in X) -.5
        elif name == "nball":
            self.f = lambda X : sum(x**2 for x in X) - .25

        ######  bounds  ##################
        bounds= kwargs.get('bounds', None) or (-1,1)
        bounds = (2*dim//len(bounds))*bounds
        self.bounds = bounds 
        
        ######  Loss function and its derivatives ##################
        loss = kwargs.get('loss', None)
        loss_prime = kwargs.get('loss_prime', None)
        if not (loss and loss_prime):
            loss_name=kwargs.get('loss_name', None)
            self.loss_name=loss_name
            if loss_name == "softplus":
                loss = lambda s :np.log(1 + np.exp(-s))
                loss_prime = lambda s : -1/(1 + np.exp(s))
            elif loss_name == "demanding":
                EPS = 0.1
                loss = lambda s : np.sqrt(EPS + (s - 1)**2) - s + 1
                loss_prime = lambda s : (s - 1)/np.sqrt(EPS + (s - 1)**2) - 1 
            else:
                EPS = 0.1
                loss = lambda s : np.sqrt(EPS + s**2) - s
                loss_prime = lambda s : s/np.sqrt(EPS + s**2) - 1
        self.loss, self.loss_prime = loss, loss_prime
        
        
    def criteria(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            f : function  (optional if 'name' is prescribed)
            name : string (optional if 'f' is prescribed)
            dim=2 : integer
            bounds=(-1,1) :tuple of 2 or 2*'dim' floats
        Does
        -------
            set pb.f, pb.dim and pb.bounds (see the description of description 
                                            of __init__)
        """
        dim = kwargs.get("dim",None) or 2
        
        bounds= kwargs.get('bounds', None) or (-1,1)
        bounds =  (2*dim//len(bounds))*bounds
        self.bounds = bounds 
        
        f =kwargs.get('f', None)
        if f:
            self.f = f
        else:
            name= kwargs.get('name', None)
            if name == "sin": 
                self.f = lambda X : X[1] - np.sin(np.pi*X[0])
            elif name == "affine":
                self.f =lambda X : 2*X[0] - X[1] - .5
            elif name == "disk":
                self.f = lambda X : X[0]**2 + X[1]**2 - .25
            elif name == "square":
                self.f = lambda X : np.maximum(np.abs(X[0]), np.abs(X[1])) -.5
            elif name == "ring":
                self.f = lambda X : 9/64 - (13/16)*(X[0]**2
                                        + X[1]**2) + (X[0]**2 + X[1]**2)**2
            elif name == "ncube":
                self.f = lambda X : max(np.abs(x) for x in X) - .5
            elif name == "nball":
                self.f = lambda X : sum(x**2 for x in X) - .25
  
    def define_loss(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            loss : numerical function : float |-> float
            loss_prime : numerical function (derivative of the former)
            or
            loss_name : string
        Does
        -------
            set pb.loss, pb.loss_prime (see the description of __init__)
        """
        loss = kwargs.get('loss', None)
        loss_prime = kwargs.get('loss_prime', None)
        if not (loss and loss_prime):
            loss_name=kwargs.get('loss_name', None)
            self.loss_name=loss_name
            if loss_name == "softplus":
                loss = lambda s :np.log(1 + np.exp(-s))
                loss_prime = lambda s : -1/(1 + np.exp(s))
            elif loss_name == "demanding":
                EPS = 0.1
                loss = lambda s : np.sqrt(EPS + (s - 1)**2) - s + 1
                loss_prime = lambda s : (s - 1)/np.sqrt(EPS + (s - 1)**2) - 1 
            else:
                EPS = 0.1
                loss = lambda s : np.sqrt(EPS + s**2) - s
                loss_prime = lambda s : s/np.sqrt(EPS + s**2) - 1
        self.loss, self.loss_prime = loss, loss_prime
            
    def show_border(self, style='r-'):
        """
        Parameters
        ----------
        style : string (plt.plot line style) 
        
        Does
        -------
        Graphical representation of the limits of the set {self.f<0}
        in the region delimited by self.bounds
        """
        name = self.name
        if name == "sin": 
            g = lambda x : .75*np.sin(np.pi*x)
            xx = np.linspace(-1, 1, 100)
            plt.plot(xx, g(xx), style)
        elif name == "affine":
            g =lambda x : 2*x - .5
            xx = np.linspace(-1,1,3)
            plt.plot(xx,g(xx),style)
        elif name == "disk":
            theta=np.linspace(0,2*np.pi,100)
            plt.plot(.5*np.cos(theta),.5*np.sin(theta),style)
        elif name == "square":
            a, b = -.5,.5
            plt.plot([a, a, b, b, a], [a, b, b, a, a], style)
        elif name == "ring":
            theta=np.linspace(0,2*np.pi,100)
            plt.plot(.5*np.cos(theta),.5*np.sin(theta),style)
            plt.plot( .75*np.cos(theta), .75*np.sin(theta),style)          
        else:
            plt.plot([0],[0],'.') 
        #
        if self.bounds: 
            (a,b,c,d)=self.bounds
            plt.axis(self.bounds) 
            plt.plot([a,b,b,a,a],[c,c,d,d,c],'--k')
            plt.axis('off')
            plt.axis('equal')
        #if self.bounds: plt.axis(self.bounds)
        
############################################################################
"""                     The class nD_data                               """
############################################################################
class nD_data:
    """
    dim, n:  two non-negative integers
    X : numpy array of shape n x dim 
    Y : numpy array of length n 
    Ypred : numpy array of length n 
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs :
            dim=2 : integer (>0)
            n : integer (>0)
            X : numpy array of shape n x dim
            Y : numpy array of length n
            f : function : numpy array of length 
            pb : object of type ToyPb
            bounds=(-1,1) : a tuple of 2 or 2 x dim floats
            init_pred = False : boolean
            
        Does
        -------
        Builds an object of the class nD_data.\
            if 'X' is given. self.X=X, (also self.Y=Y)
            if 'X' is not given but n is given, an array self.X is created
            of shape n x dim made of n random vectors drawn in the box given 
            by bounds or pb.bounds or (-1,1)*dim
            
            if 'Y' is not given then f or pb.f is used to build
            self.Y as self.Y[j] = sign(f(self.X[j,:]))
            
            if init_pred : creates self.Ypred, a zero numpy array of length n  
        """
        X = kwargs.get("X", None)
        self.Y = None
        pb = kwargs.get("pb", None)
        try:  
            self.n, self.dim = X.shape
            self.X = X
            self.Y = kwargs.get("Y", None)
        except:    
            n = kwargs.get("n", None)
            if n:
                self.n = n
                dim = kwargs.get("dim", None) or (pb and pb.dim) or 2
                bounds = kwargs.get("bounds",
                                    None) or (pb and pb.bounds) or (-1,1)
                bounds= bounds*(2*dim//len(bounds))
                X = nprd.rand(n, dim)
                for k in range(dim):
                    X[:, k] = bounds[2*k] + (bounds[2*k + 1] 
                                               - bounds[2*k])*X[:, k]
                self.X, self.bounds, self.dim  = X, bounds, dim
        f=kwargs.get("f")
        if self.n and (f or pb) and self.Y == None :
            f = f or pb.f                     
            n, X, Y  = self.n, self.X, np.zeros(n)
            for i in range(n):
                Y[i] = 2*(f(X[i]) > 0) - 1
            self.Y = Y            
            
        if self.n and kwargs.get("init_pred"):
            self.Ypred = np.zeros(self.n)
        
    def prediction(self, nn, A, zero_one=False):
        """
        Parameters
        ----------
        nn : object of type ToyNN
        A : coefficients corresponding to NN
        zero_one=False : boolean
        
        Does
        -------
        Builds or updates (if exists) self.Ypred with the outputs of the 
        coefficients of A with inputs self.X[j], j=0,...
        
        sign = sign or zero_one
        if zero_one : 
            self.Ypred[j] = sign( NN.output(A,self.X[j]))
        else :
            self.Ypred[j] =  NN.output(A,self.X[j])
        """  
        try : Ypred = self.Ypred
        except :
            self.Ypred = np.zeros([self.n])
            Ypred = self.Ypred
        if zero_one:
            for j in range(self.n): 
                Ypred[j] = 2*(nn.output(A,self.X[j]) > 0) - 1  
        else: 
            for j in range(self.n): Ypred[j] = nn.output(A, self.X[j]) 
    
    def init_pred(self):
        """
        Intialize self.Ypred to 0 (useless function)
        """
        self.Ypred = np.zeros(self.n)
            
    def classify(self,**kwargs):
        """
        Parameters
        ----------
        **kwargs
            f : function : float |-> float
            pb : object of type ToyPb

        Does
        -------
        builds or updates self.Y  with f or pb.f
        self.Y[j] = sign( f(self.X[j]) )
        """
        f = kwargs.get("f",None) or kwargs.get("pb",None).f
        try: Y = self.Y
        except: 
            self.Y = np.zeros(self.n)
            Y = self.Y
        for i in range(self.n): Y[i] = 2*(f(self.X[i]) > 0) - 1
    
    def show_class(self, pred = False, wrong_only=None):
        """
        Parameters
        ----------
        pred = None : boolean
        wrong_only = None : boolean
        
        Does
        -------
        Graphic representation of points self.X[j] with different colors 
        according to the sign of self.Y[j] (or self.Ypred[j] if 'pred'==True)
        
        If wrong_only : shows only misclassified points 
                        (X[j] with Y[j]*Ypres[j]<0).
        """
        if pred:
            X, Y = self.X, self.Ypred
        else:
            X, Y = self.X, self.Y
        if wrong_only:
            ind = np.where(self.Y*Y < 0)
            plt.scatter(X[ind,0], X[ind,1], c='k', s=3, label="misclass.")
        else:
            ind = np.where(Y>0)
            plt.scatter(X[ind,0], X[ind,1], c='b', s=3, label=r"$y=1$")
            ind  = np.where(Y<0)
            plt.scatter(X[ind,0], X[ind,1], c='r', s=3, label=r"$y=-1$")
        plt.axis(self.bounds)      
        plt.axis('off')

############################################################################
"""                     The class ToyNN                                  """
############################################################################
class ToyNN:
    """ 
    class of simple Neural Networks 
      - N : an integer. N - 1 is the number of hidden layers of the neural 
              network, so N is at least 1. 
      - card : a list of int which represent the number of nodes in each layer, 
            Its length is at least 2. It starts with a 2 and ends with a 1, 
            [2, a1, a2, ..., 1]                    
      - Nparam : integer. The number of parameters in the NN
      - chi : numerical function of 1 variable (activation function)
      - chi_prime : numerical function of 1 variable (derivative of the former) 
      - xx, yy, zz: 3 2D arrays for graphic representations of the NN's output 
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs : 
            card: list or tuple of integers (number of neurons in each layer).
            coef_bounds: None or tuple of 4 floats.
            chi='tanh' : float|->float numeric function (activation function)
            chi_prime : loat|->float numeric function (derivative of chi)
            grid : list or tuple (see description of init_grid below). 
                        If grid = None self.xx, self.yy, self.zz are 
                        not created. 
                        
        Does
        -------
        Builds a description and tools for NN :
            the structure is given by the parameter card.
            the range of the coefficients [W, Bias] created by the method creat_rand(self)
            are specified by coef_bound
            if coef_bounds == None : coef_bounds is set to (-1,1,-1,1)
            if coef_bounds == (a,b,c,d) : the coefficients W and Bias and self.Bias are random 
                numpy arrays with 
                a ≤ self.W[n][i,j] ≤ b,  c ≤ self.Bias[n][i,j] ≤ d 
        
        Remark
        -------
        An element of the class ToyNN only describes the structure of a NN
        (nb of nodes, activation function, ...) but does not contain the 
        coefficients of a specific NN of this form.
        These latter are created and stored as separate objects 
        In the methods below such objects are denoted A, B or C  
        For instance A is the list [A_W, A_Bias] where:
            - A_W is a list of N 2D numpy arrays. A_W[n][i,j] is the weight 
            of the edge between the iest node of layer n and the jiest 
            node of layer n + 1.  
            - A_Bias is a list of N 1D numpy arrays. A_Bias[n][i] is the 
            bias on node i of layer n + 1 
        """
        card = kwargs.get("card", None)
        self.card = card
        N = len(card) - 1
        self.N = N
        Nparam = 0    # Nparam is the number of parameters to be optimized
                      # (nbr of entries of W + nbr of entries of Bias) 
        for n in range(N):
            Nparam += (card[n] + 1)*card[n + 1]
        self.Nparam = Nparam
            
        chi = kwargs.get("chi", None)    
        if chi == "tanh":
            self.chi = lambda x : np.tanh(x)
            self.chi_prime = lambda x : 1/np.cosh(x)**2
        elif chi == "sigmoide":
            self.chi = lambda x : 1/(1 + np.exp(-x))
            self.chi_prime = lambda x : np.exp(x)/(1 + np.exp(x))**2
        elif chi == "RELU":
            self.chi = lambda s : np.where(s > 1, s - 1, 0)
            self.chi_prime = lambda s : np.where(s > 1, 1, 0)
        elif chi:
            self.chi=chi
            self.chi_prime =  kwargs.get("chi_prime", None) 
        
        coef_bounds = kwargs.get("coef_bounds", None)
        self.coef_bounds = coef_bounds or (-1,1,-1,1)

        grid = kwargs.get("grid", None)  
        if grid:
            if len(grid) == 3:
                a, b, c, d, nx, ny = (grid[0], grid[1],grid[0], grid[1],
                                      grid[2], grid[2])
            elif len(grid) == 6:
                a, b, c, d, nx, ny = (grid[0], grid[1], grid[2], grid[3],
                                      grid[4], grid[5])            
            x=np.linspace(a, b ,nx)
            y=np.linspace(c, d, ny)
            self.xx, self.yy = np.meshgrid(x,y)
            self.zz = np.zeros(self.xx.shape)
                 
            
    def create_zero(self):
        """
        Parameters
        ----------

        Does 
        -------
        create lists of numpy arrays dW and dBias with null coefficients.
        returns a list [W, Bias] where W and Bias are lists of 
        numpy arrays
        """
        N, card = self.N, self.card
        W, Bias =[],[]
        for n in range(N):
            W.append(np.zeros([card[n], card[n + 1]]))
            Bias.append(np.zeros(card[n + 1]))
        return [W, Bias]
    
    
    def create_rand(self, coef_bounds=None):
        """
        Parameters
        ----------
        coef_bounds = None : list of 4 floats

        Does
        -------
        Creates a list [W,Bias] of two lists of numpy arrays
        with random values:coefficients 
        a ≤ W[i][j,k] ≤ b,   c ≤ Bias[i][j,k] ≤ d
        with 
        a,b,c,d= coef_bounds or self.coef_bounds
        """
        a, b, c, d = coef_bounds or self.coef_bounds
        W, Bias=[], []
        card = self.card
        for n in range(self.N):
            W.append((b - a)*nprd.rand(card[n],card[n + 1]) + a)
            Bias.append((d - c)*nprd.rand(card[n + 1]) + c)
        return [W, Bias]
    
    def copy(self, A):
        """
        Parameters
        ----------
        A : list of 2 lists of numpy arrays
        
        Returns
        -------
        copy of A
        """
        W, Bias = A[0], A[1]
        W_, Bias_ = [], []
        for n in range(self.N):
            W_.append(W[n].copy())
            Bias_.append(Bias[n].copy())
        return [W_, Bias_]
    
    def add(self,A,B,c=None,output=True):
        """
        Parameters
        ----------
        A, B : lists of lists of numpy arrays
        c  : float (c=None <=> c=1)
        output : boulean
        
        Does  
        -------
        compute C = A+c*B. 
        if output == true return C 
        else place the result in A

        """
        A_W, A_Bias = A[0], A[1]
        B_W, B_Bias = B[0], B[1]  
        if output:
            W, Bias = [],[]
            if c==None:
                for n in range(self.N):
                    W.append(A_W[n] + B_W[n])
                    Bias.append(A_Bias[n] + B_Bias[n])
            else:
               for n in range(self.N):
                   W.append(A_W[n] + c*B_W[n])
                   Bias.append(A_Bias[n] + c*B_Bias[n])
            return [W, Bias]
        else:
            if c==None:
                for n in range(self.N):
                    A_W[n] += B_W[n]
                    A_Bias[n] += B_Bias[n]
            else:
                for n in range(self.N):
                    A_W[n] += c*B_W[n]
                    A_Bias[n] += c*B_Bias[n]
                                        
    def scal_mult(self,A,c, output=True):                    
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
        c : float
        output : boolean (default is True)
        Does  
        -------
        Computes c*A and returns the result if output=True 
        if output == True, or put the resut in A in the other case 
        
        """
        A_W, A_Bias = A[0], A[1]
        if output:
            W, Bias = [], []
            for n in range(self.N):
                W.append(c*A_W[n])
                Bias.append(c*A_Bias[n])
            return [W, Bias]
        else:
            for n in range(self.N):
                A_W[n]*=c
                A_Bias[n]*=c
                
    def count_non_zero(self,A):                    
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
          
        Does  
        -------
        return the number of non zero coefficients of A
        """
        W, Bias = A[0], A[1]
        s = 0
        for n in range(self.N):
            s += np.sum(np.where(W[n]==0,0,1))
            s += np.sum(np.where(Bias[n]==0,0,1))
        return s
                               
    def dot(self, A, B):
        """
        Parameters
        ----------
        A, B : lists of lists of numpy arrays

        Returns
        -------
        dot product of A and B
        """ 
        A_W, A_Bias = A[0], A[1]
        B_W, B_Bias = B[0], B[1]  
        s= 0
        for n in range(self.N):
            s += np.sum(A_W[n] * B_W[n]) + np.sum(A_Bias[n] * B_Bias[n])
        return s

    def square(self, A, output=True):
        """
        Parameters
        ----------
        A : list of 2 lists of numpy arrays
        output=True : boolean
        
        Does
        -------
        Computes the square of the coefficients of A
        and returns it in the same format as A if output = True. 
        If not output, the result is put in A  
        """
        A_W, A_Bias = A[0], A[1]
        if output:
            W, Bias = [],[]
            for n in range(self.N):
                W.append(A_W[n]**2)
                Bias.append(A_Bias[n]**2)
            return [W, Bias]
        else:
            for n in range(self.N):
                A_W[n] = A_W[n]**2
                A_Bias[n] = A_Bias[n]**2  
                    
    def maps(self, f, A, output=True,param=None):
        """
        Parameters
        ----------
        f : function which takes a numer and a list of parameters    
        A : list of 2 lists of numpy arrays
        output=True : boolean
        param=None : anything (third parameter of f)
        
        Does
        -------
        computes f(A,l_param) and returns it if output = True. If not output, 
        the result is put in A  
        """
        A_W, A_Bias = A[0], A[1]
        if output:
            W, Bias = [],[]
            if param:
                for n in range(self.N):
                    W.append(f(A_W[n], param))
                    Bias.append(f(A_Bias[n], param))
            else:
                for n in range(self.N):
                    W.append(f(A_W[n]))
                    Bias.append(f(A_Bias[n]))
            return [W, Bias]
        else:
            if param:
                for n in range(self.N):
                    A_W[n] = f(A_W[n], param)
                    A_Bias[n] = f(A_Bias[n], param)
            else:
                for n in range(self.N):
                    A_W[n] = f(A_W[n])
                    A_Bias[n] = f(A_Bias[n])          
                    
    def maps2(self, f, A, B, output=True,param=None):
        """
        Parameters
        ----------
        f : function which takes a numer and a list of parameters    
        A, B : lists of 2 lists of numpy arrays
        output=True : boolean
        param=None : anything (third parameter of f)
        
        Does
        -------
        computes f(A,B,l_param) and returns it if output = True. If not output, 
        the result is put in A  
        """
        A_W, A_Bias = A[0], A[1]
        B_W, B_Bias = B[0], B[1]        
        if output:
            W, Bias = [],[]
            if param:
                for n in range(self.N):
                    W.append(f(A_W[n], B_W[n], param))
                    Bias.append(f(A_Bias[n], B_Bias[n], param))
            else:
                for n in range(self.N):
                    W.append(f(A_W[n], B_W[n]))
                    Bias.append(f(A_Bias[n], B_Bias[n]))
            return [W, Bias]
        else:
            if param:
                for n in range(self.N):
                    A_W[n] = f(A_W[n], B_W[n], param)
                    A_Bias[n] = f(A_Bias[n], B_Bias[n], param)
            else:
                for n in range(self.N):
                    A_W[n] = f(A_W[n], B_W[n])
                    A_Bias[n] = f(A_Bias[n], B_Bias[n])           
      
    
    # Forward computation of the output value for a data 
    def output(self, A, x):
        """
        Parameters
        ----------
        A : list of 2 lists of numpy arrays
        X = [a,b, ...] with a,b,... float
             or numpy array

        Returns
        -------
        the output of the neural network with coefs A 
        with the input X

        """
        W, Bias = A[0], A[1]
        o = np.array(x)
        for n in range(self.N -1):
            i = W[n].T.dot(o) + Bias[n]
            o = self.chi(i)
        return  (W[-1].T.dot(o) + Bias[-1])[0]

    # Gradient of output with respect to W and Bias (multplied by -tau)
    def descent(self, A, x, y, alpha=None, add= False, B=None,  **kwargs):
        """    
        Parameters
        ----------
        A : list of 2 lists of numpy arrays
        X = [a,b, ...] with a,b,... float
                or numpy array
        y : float
        alpha=None : float   (optional)
        add=False : boolean
        B : list of 2 lists of numpy arrays
        **kwargs 
            f : function f:float|->float given or = 'pb'.loss_prime
            pb :  object of type ToyPb
                either f or pb have to be given
            
        Does
        -------
        Computes the opposite of the gradient of F(y*self.output(A,X))
        denoted dA
        with respect to the coefficients A.
        (see the description of self.output(A,X) above)
        f or pb.f is the derivative of F.
        
        If alpha != None: multiply dA by alpha
        if B != None:
            return dA
        else:
            add dA to B 
        """
        
        # Forward computation of input and output values at the nodes 
        A_W, A_Bias =A[0], A[1]
        o = np.array(x)
        O, I =[], [o]    
        for n in range(self.N):
            O.append(o)      
            i = A_W[n].T.dot(o) + A_Bias[n]
            I.append(i)
            o = self.chi(i)
            
        # Backward computation of the gradients
        pb, N = kwargs.get("pb"), self.N
        alpha = alpha or 1
        f = kwargs.get("f") or (pb and pb.loss_prime) or 1  
        if f==1:
            desc_bias = np.array([1.])
        else:
            desc_bias = -alpha*y*f(y*i)
        if add:
            for n in range(N - 1, 0, -1):
                A_Bias[n] += desc_bias
                A_W_n = A_W[n].copy()
                A_W[n] += np.tensordot(O[n], desc_bias, 0)
                desc_bias = self.chi_prime(I[n])*(A_W_n.dot(desc_bias))
            A_Bias[0] += desc_bias
            A_W[0] += np.tensordot(O[0], desc_bias, 0)
        elif B:
            W, Bias =B[0], B[1]
            for n in range(N - 1, 0, -1):
                Bias[n] += desc_bias
                W[n] += np.tensordot(O[n], desc_bias, 0)
                desc_bias = self.chi_prime(I[n])*(A_W[n].dot(desc_bias))
            Bias[0] += desc_bias
            W[0] += np.tensordot(O[0], desc_bias, 0)
        else:
            dW, dBias = [],[]
            for n in range(N - 1, 0, -1):
                dBias.append(desc_bias)
                dW.append(np.tensordot(O[n], desc_bias, 0))
                desc_bias = self.chi_prime(I[n])*(A_W[n].dot(desc_bias))
            dBias.append(desc_bias)
            dW.append(np.tensordot(O[0], desc_bias, 0))
            dW.reverse();  dBias.reverse()
            return [dW, dBias]
  
    
    def def_grid(self, grid):
        """
        Parameters
        ----------
        grid : list or tuple of float and integers
        
        Does
        ------
        Creates a grid self.xx, self.yy, self.zz
        These 3 objects are nx x ny numpy arrays,
        self.xx contains the abscissas, self.yy the ordinates
        self.zz is a zero numpy array.
        (this grid is used to compute 
         self.zz[i,j] = self.output(self.xx[i,j],self.yy[i,j])
         in function show_pred )
        """
        if len(grid) == 3:
            a, b, c, d = grid[0], grid[1],grid[0], grid[1]
            nx, ny = grid[2], grid[2]
        elif len(grid) == 6:
            a, b, c, d = grid[0], grid[1], grid[2], grid[3]
            nx, ny = grid[4], grid[5]               
        x=np.linspace(a,b,nx)
        y=np.linspace(c,d,ny)
        self.xx, self.yy = np.meshgrid(x,y)
        self.zz = np.zeros(self.xx.shape)
                       
    def def_activation(self, chi = None, chi_prime= None):
        """
        Parameters
        ----------
        chi = None : numerical function of 1 variable (activation function)
        chi_prime = None : numerical function of 1 variable 
                            (derivative of the former)

        Does
        -------
        Set the activation function self.chi and its derivative self.chi_prime

        """
        if chi == "sigmoide":
            self.chi = lambda x : 1/(1 + np.exp(-x))
            self.chi_prime = lambda x : np.exp(-x)/(1 + np.exp(-x))**2
        elif chi == "RELU":
            self.chi = lambda s : np.where(s > 1, s - 1, 0)
            self.chi_prime = lambda s : np.where(s > 1, 1, 0)            
        elif chi:
            self.chi=chi
            self.chi_prime = chi_prime
        else:
            self.chi = lambda x : np.tanh(x)
            self.chi_prime = lambda x : 1/np.cosh(x)**2


    ### Computation of predictions (arrays of outputs) 
    def prediction(self, A, data, zero_one=False):
        """
        Parameters
        ----------
            A : list of list of numpy arrays    
            DATA : object of type nD_data
            zero_one = False : boolean 

        Does
        -------
        Updates the numpy array data.Ypred.
        if not zero_one:
            Ypred[j] = self.output(A,data.X[j])
        if sign :
            Ypred[j] = sign of self.output(A,data.X[j])
        """
        X, n = data.X, data.n     
        try : Ypred = data.Ypred
        except :
            data.Ypred=np.zeros(np.shape(X)[0])
            Ypred=data.Ypred 
        if zero_one:
            for j in range(n): Ypred[j] = 2*(self.output(A,X[j]) > 0) - 1  
        else: 
            for j in range(n): Ypred[j] = self.output(A,X[j]) 
                
    #### total loss  
    def total_loss(self, A, data, zero_one=False, **kwargs):
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
        data : type nD_data
        zero_one=False : boolean
        **kwargs :
            pb : object of type problem
            loss : float |-> float numeric function 
          
        Return
        --------
        Return the average loss for the prediction of A on the data DATA
        that is : 
            (1/n) sum loss( Y[j]*self.output(A,X[j])) 
        where X,n,Y = data.X, data.n, data.Y 
        The loss function used is 
          0/1 if zero_one is set to True
          or loss if loss is precribed
          or pb.loss if pb is prescibed
          or soft_plus ( loss(s) = log(1 + exp(-3*s)) )
        """
        X, Y, n, cost = data.X, data.Y, data.n, 0
        if zero_one:
            for j in range(n):
                cost += Y[j]*self.output(A, X[j]) < 0
        else :
            loss = kwargs.get("loss",None) or kwargs.get("pb", 
                            None).loss or (lambda s :np.log(1 + np.exp(-3*s)))
            for j in range(n): 
                cost += loss(Y[j]*self.output(A, X[j]))
        return cost/n
        
    ### total loss and prediction 
    def total_loss_and_prediction(self, A, data, zero_one=False, **kwargs):
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
        data : type nD_data
        zero_one=False : boolean
        **kwargs :
            pb : object of type problem
            loss : float |-> float numeric function 
            
        Does
        -------
        Updates the numpy array data.Ypred like the function prediction above.
        if not zero_one:
            Ypred[j] = self.output(A,data.X[j])
        if zero_one :
            Ypred[j] = sign of self.output(A,data.X[j])
        
        Returns
        --------
        The average loss for the prediction of self on the data set "data" like 
        the function total_loss above.
        """
        X, n, Y = data.X, data.n, data.Y
        try : Ypred = data.Ypred
        except :
            data.Ypred=np.zeros(np.shape(X)[0])
            Ypred=data.Ypred
        cost = 0
        if zero_one:
            for j in range(n):
                ypred = 2*(self.output(A, X[j])>0) - 1
                cost +=  ypred*Y[j] < 0
                Ypred[j] = ypred
        else: 
            loss = kwargs.get("loss",None) or kwargs.get("pb", 
                            None).loss or (lambda s :np.log(1 + np.exp(-3*s)))
            for j in range(n):
                ypred = self.output(A, X[j]) 
                cost += loss(Y[j]*ypred)
                Ypred[j] = ypred
        return cost/n
    
    ### Displays the classification realized by self on the points of the grid
    def show_pred(self, A):
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
        
        Does
        -------
        Displays zz[i,j]=self.output([self.xx[i,j], self.yy[i,j]]) on the 
        grid self.xx, self.yy
        (zz contains the predictions of self on the grid points)
        """
        nx,ny = np.shape(self.xx)
        for i in range(nx):
            for j in range(ny):
                    self.zz[i,j] = self.output(A,[self.xx[i,j], self.yy[i,j]])
        plt.imshow(self.zz,
                   interpolation='bilinear', origin='lower',cmap=cm.RdYlBu, 
                   vmin=-1, vmax=1,alpha =.6, extent=(-1, 1, -1, 1))
        plt.axis('off')
        
    ### Function for the graphical representation of the NN
    def show(self, A):
        """
        Parameters
        ----------
        A : list of lists of numpy arrays
        
        Does
        -------
        Displays the neural network 'A=[W,Bias]'. The widths of the links 
        reflect the magnitude of the coefficients W[n][i,j]. The color of 
        the links and of the neurons depend on the sign of the coefficients
        W[n][i,j] and Bias[n][i].
        """
        W, Bias = A[0], A[1]
        xright = 0
        nright = self.card[0]
        dl = .5/nright
        HR=np.linspace(dl,1-dl,nright)
        xright = 0
        for n in range(self.N):
            w = W[n]
            xleft = xright
            xright += np.sqrt(w.size)
            nleft, nright = nright, w.shape[1]
            dl = .5/nright
            HL, HR = HR, np.linspace(dl,1-dl,nright)
            MAX = np.amax(abs(w)) + 1e-12
            colors = ["tomato", "teal"]  # 0: negative, 1: positive
            for j in range(nleft):
                for k in range(nright):
                    val = w[j,k]
                    plt.plot([xleft, xright], [HL[j], HR[k]],
                             color=colors[val>0],
                             lw=1.5, alpha=abs(val)/MAX, zorder=5)
            if n:
                b = Bias[n - 1]             
                bcolor = np.where(b<0,colors[0],
                                  np.where(b>0,colors[1], 'white'))
                plt.scatter([xleft]*nleft,HL, ec='dimgrey', 
                        fc=bcolor, s=350, lw=1.5, zorder=10)
            else: plt.scatter([xleft]*nleft,HL, ec='dimgrey',
                              fc='white', s=350, lw=1.5, zorder=10)
        b=Bias[n]        
        bcolor= colors[0] if b<0 else colors[1] if b>0 else 'white'
        plt.scatter([xright]*nright,HR, color='dimgrey', 
                        fc=bcolor, s=350, lw=1.5, zorder=10)
        plt.axis('off')
        plt.show()  

############################################################################### 
"""                             EOF                                         """
###############################################################################