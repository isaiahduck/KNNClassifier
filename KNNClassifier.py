import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# Returns xTr,yTr,xTe,yTe
# xTr, xTe are in the form nxd
# yTr, yTe are in the form nx1
def loaddata(filename):
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T

#LOAD DATA
xTr,yTr,xTe,yTe=loaddata("faces.mat") 

def plotfaces(X, xdim=38, ydim=31, ):
    n, d = X.shape
    f, axarr = plt.subplots(1, n, sharey=True)
    f.set_figwidth(10 * n)
    f.set_figheight(n)
    
    if n > 1:
        for i in range(n):
            axarr[i].imshow(X[i, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
    else:
        axarr.imshow(X[0, :].reshape(ydim, xdim).T, cmap=plt.cm.binary_r)
plotfaces(xTr[:10, :])

# function innerproduct(X,Z)
# Computes the inner-product matrix.
# Syntax:
# D=innerproduct(X,Z)
# Input:
# X: nxd data matrix with n vectors (rows) of dimensionality d
# Z: mxd data matrix with m vectors (rows) of dimensionality d
# Output:
# Matrix G of size nxm
# G[i,j] is the inner-product between vectors X[i,:] and Z[j,:]
# call with only one input:
# innerproduct(X)=innerproduct(X,X)
def innerproduct(X,Z=None):
    if Z is None: 
        Z = X
    G = np.matmul(X, np.transpose(Z))
    return G

# function D=l2distance(X,Z)
#
# Computes the Euclidean distance matrix.
# Syntax:
# D=l2distance(X,Z)
# Input:
# X: nxd data matrix with n vectors (rows) of dimensionality d
# Z: mxd data matrix with m vectors (rows) of dimensionality d
#
# Output:
# Matrix D of size nxm
# D(i,j) is the Euclidean distance of X(i,:) and Z(j,:)
#
# call with only one input:
# l2distance(X)=l2distance(X,X)
def l2distance(X,Z=None):
    if Z is None:
        (n,d1) = X.shape
        G = innerproduct(X)
        S = np.repeat(G.diagonal(), n)
        S = S.reshape(n, n)
        R = S.transpose()
    else:
        G = innerproduct(X, Z)
        S = innerproduct(X)
        R = innerproduct(Z)
        (n,d1) = X.shape
        (m, d2) = Z.shape
        S = np.repeat(S.diagonal(), m)
        R = np.repeat(R.diagonal(), n)
        S = S.reshape(n, m)
        R = np.transpose(R.reshape(m, n))
    return np.sqrt(S - 2*G + R)

# function [indices,dists]=findknn(xTr,xTe,k);
# Finds the k nearest neighbors of xTe in xTr.
# Input:
# xTr = nxd input matrix with n row-vectors of dimensionality d
# xTe = mxd input matrix with m row-vectors of dimensionality d
# k = number of nearest neighbors to be found
# Output:
# indices = kxm matrix, where indices(i,j) is the i^th nearest neighbor of xTe(j,:)
# dists = Euclidean distances to the respective nearest neighbors
def findknn(xTr,xTe,k):
    m,d = xTe.shape
    I = np.zeros((k, m), dtype = np.int)
    D = np.zeros((k, m))
    eucD = l2distance(xTe, xTr)
    
    for x in range(m): 
        arr = eucD[x, :]
        sortArr = np.argsort(arr)[:k]
        I[:,x] = sortArr
        D[:,x] = np.take(arr, sortArr)

    return I, D

# function output=analyze(kind,truth,preds)         
# Analyses the accuracy of a prediction
# Input:
# kind='acc' classification error
# kind='abs' absolute loss
# (other values of 'kind' will follow later)
def analyze(kind,truth,preds):   
    truth = truth.flatten()
    preds = preds.flatten()
    
    diff = np.absolute(truth - preds)
    sumDiff = np.sum(diff)
    
    if kind == 'abs':
        output = sumDiff/len(diff) 
    elif kind == 'acc':
        if sumDiff == 0:
            output = 1
        else:
            output = (float(1) - (np.count_nonzero(diff))/float(len(diff)))     
    return output

# function preds=knnclassifier(xTr,yTr,xTe,k);
# k-nn classifier 
# Input:
# xTr = nxd input matrix with n row-vectors of dimensionality d
# xTe = mxd input matrix with m row-vectors of dimensionality d
# k = number of nearest neighbors to be found
# Output:
# preds = predicted labels, ie preds(i) is the predicted label of xTe(i,:)
def knnclassifier(xTr,yTr,xTe,k):
    m,d = xTe.shape
    I,D = findknn(xTr, xTe, k)
    preds = np.zeros(m)
    
    for c in range(m):
        classifications = np.take(yTr, I[:, c]).astype(int)
        preds[c] = np.argmax(np.bincount(classifications)
    return preds