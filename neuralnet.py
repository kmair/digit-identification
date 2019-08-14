 # -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:31:51 2019

@author: Kanishk
"""

import sys
import numpy as np
import csv


def Data_load(inp):
    """Creates dataset. 1st column is Y while rest of the columns are X data.
    After reading the input file "inp", we make Y and X files where:
    Y and X are lists. Each line of X is stored as an array""" 

    with open(inp) as tsvip:
        reader = csv.reader(tsvip, delimiter = ',')
        Y = []
        X = []
        for i, row in enumerate(reader):
            label = int(row[0])
            Y.append(label)
            x_data = row[1:] 
            M = len(x_data)
            temp = []
            for j in range(M):
                temp.append(int(x_data[j]))
            
            X.append(np.array(temp))

    return X, Y

def Ydata_initial(ylist):
    '''Creates a sparse matrix where we get a matrix of dimensions: (points, labels).
    This is done to aid in softmax'''

    unique_list = []
    for x in ylist: 
        if x not in unique_list: 
            unique_list.append(x)
    
    k = len(unique_list)
    Datapts = len(ylist)
    y = np.zeros((Datapts,k))
    
    for dp in range(Datapts):
        index = ylist[dp]
        y[dp, index]=1
    return y, k

def Ini_wt(r, c, Flag): 
    alp = np.zeros((r,c))  
    "c is M+1 i.e. counting the bias term too!"
    if Flag == 1:
        for j in range(r):
            alp[j] = np.random.uniform(-0.1,0.1,c)
            alp[j,0] = 0
    return alp

# Forward:
def Lin_fwd(x, theta):
    "x doesn't contain x0 = 1 term. Theta has bias term"
    ini = np.array([1])
    X = np.concatenate((ini, x))
    lin = X@theta.T 
    return lin
   
def Sigm_fwd(a):
    sig = 1/(1+np.exp(-a))    
    return sig

def Soft_fwd(b):
    Nr = np.exp(b)
    Dr = sum(Nr)
    return Nr/Dr

def Cross_Entropy_Fwd(yh, y):
    lyh = np.log(yh)
    return -np.dot(y, lyh)
    
def NNForward(x,y,alpha,beta):
    a = Lin_fwd(x, alpha)
    z = Sigm_fwd(a)
    b = Lin_fwd(z, beta)
    yhat = Soft_fwd(b)
    J = Cross_Entropy_Fwd(yhat, y)
    return a, z, b, yhat, J

# Backward
def Cross_Entropy_Bwd(yhat, y, gJ=1):
    gyhat = -y/yhat *gJ                 
    return gyhat 

def Soft_Bwd(yhat, gyhat):
    K = len(yhat)
    yh = np.reshape(yhat, (K,1))
    dyhatdb = np.eye(K)*yhat - yh@yh.T
    dLdb = dyhatdb@gyhat
    dLdb = np.reshape(dLdb, (len(dLdb),1))
    return dLdb

def Lin_Bwd(z, beta, gb):  # equivalent to (x, alpha, ga)
    # gBeta
    ini = np.array([1])
    Z = np.concatenate((ini, z))
    dbdB = Z           
    dbdB = np.reshape(dbdB, (len(dbdB),1))
    dLdB =  gb@dbdB.T
    
    #gZ 
    dbdz = beta[:,1:]
    dLdz = dbdz.T@gb
    
    return dLdB, dLdz

def Sig_Bwd(z, gz):
    Z = np.reshape(z, (len(z), 1))
    dzda = Z*(1-Z)
    return dzda*gz

def NNBackward(x,y,Alp,Bet,a,b,z,yhat,J):
    '''There is 1 hidden layer followed by a softmax function. To obtain their gradients in backprop, we collect the gAlp and gBet returned'''
    gyhat = Cross_Entropy_Bwd(yhat, y)
    gb = Soft_Bwd(yhat, gyhat)
    gBet,gz = Lin_Bwd(z, Bet, gb)
    ga = Sig_Bwd(z, gz)
    gAlp,gx = Lin_Bwd(x, Alp, ga)
    
    return gAlp, gBet

def SGD(Train, Test, NOE):
    '''Implements the Stochastic Gradient Algorithm to train the dataset
    NOE: number of epochs'''

    x_tr, y_tr = Data_load(Train)
    x_te, y_te = Data_load(Test)
    
    Y_train, K = Ydata_initial(y_tr)   # Array
    Y_test, _ = Ydata_initial(y_te)
    N_Tr, M = np.shape(x_tr)        # N_Tr = No. of train data pts.
    N_Te = len(y_te)                # N_Te = No. of test data pts.
    Alpha0 = Ini_wt(D, M+1, flag) 
    Beta0 = Ini_wt(K, D+1, flag)
    
    CE_train = []
    CE_test = []
    for E in range(NOE):
        for N in range(N_Tr):
            X = x_tr[N]
            Y = Y_train[N]
            a, z, b, yhat, J = NNForward(X, Y, Alpha0, Beta0)
            Ga, Gb = NNBackward(X, Y,Alpha0,Beta0,a,b,z,yhat,J)

            Alpha = Alpha0 - lr*Ga
            Beta = Beta0 - lr*Gb
            Alpha0 = Alpha
            Beta0 = Beta
        
        J_train = []
        for N in range(N_Tr):
            X = x_tr[N]
            Y = Y_train[N]
            a, z, b, yhat, J_Tr = NNForward(X, Y, Alpha0, Beta0)
            J_train.append(J_Tr)
        
        J_test = []
        for N in range(N_Te):
            X = x_te[N]
            Y = Y_test[N]
            a, z, b, yhat, J_Te = NNForward(X, Y, Alpha0, Beta0)
            J_test.append(J_Te)            
            
        Jtravg = sum(J_train)/N_Tr
        Jteavg = sum(J_test)/N_Te
        CE_train.append(Jtravg)
        CE_test.append(Jteavg)
         
    return Alpha0, Beta0, CE_train, CE_test

def Prediction(X, Y, Alpha, Beta):
    N_DP = len(Y)
    prediction = []
    for i in range(N_DP):
        a, z, b, yhat, J = NNForward(X[i], Y[i], Alpha, Beta) 
        Pred = np.argmax(yhat)
        prediction.append(Pred)
    return prediction
        
def error(Data, Alpha, Beta):
    X, Y = Data_load(Data)
    pred = Prediction(X, Y, Alpha, Beta)
    N_DP = len(Y)
    correct = 0
    for i in range(N_DP):
        if pred[i]==Y[i]:
            correct += 1   
    err = 1 - correct/N_DP
    return err

def output_metrics(opmetrics, CE_train, CE_test):
    outfile=open(opmetrics,"w+")    
    for k in range(noe):
        outfile.write("epoch=%i" %(k+1) + " crossentropy(train): %s" %(CE_train[k]) + "\n")
        outfile.write("epoch=%i" %(k+1) + " crossentropy(test): %s" %(CE_test[k]) + "\n")
    outfile.write("error(train): %g" %(train_err) + "\n")  
    outfile.write("error(test): %g" %(test_err))          
    outfile.close()
    
def output_labels(Data, Alpha, Beta, op_filename):
    X, Y = Data_load(Data)
    pred = Prediction(X, Y, Alpha, Beta)
    outfile=open(op_filename,"w+")
    for i in range(len(pred)):
        outfile.write("%g" %(pred[i]) + "\n")
        
    outfile.close()
        
if __name__ == '__main__':
    train_ip = sys.argv[1]
    test_ip = sys.argv[2]
    train_op = sys.argv[3]
    test_op = sys.argv[4]
    opmetrics = sys.argv[5]
    Number_of_epochs = sys.argv[6]
    Hidden_units = sys.argv[7]  # Hidden units
    Init_flag = sys.argv[8]
    Lambda = sys.argv[9]
    D = int(Hidden_units)
    flag = int(Init_flag)
    noe = int(Number_of_epochs)
    lr = float(Lambda)

    # Get train data
    x_tr, y_tr = Data_load(train_ip)
    Y_tr = np.array(y_tr)

    # Obtain Alpha and Beta values trained by the SGD algorithm
    Alpha, Beta, CE_train, CE_test = SGD(train_ip, test_ip, noe)

    # Compute error on the test and train Dataset
    train_err = error(train_ip, Alpha, Beta)
    test_err = error(test_ip, Alpha, Beta)
    
    # Creating the output files
    output_metrics(opmetrics, CE_train, CE_test)
    output_labels(train_ip, Alpha, Beta, train_op)       
    output_labels(test_ip, Alpha, Beta, test_op)       