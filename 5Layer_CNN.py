#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tqdm import tqdm,trange
from random import shuffle
import os
from math import log

TRAIN_DIR = #YOUR PATH TO CATS N DOGS DATA SET HERE
IMG_SIZE = 64

def label_img(img):
    lab = img.split('.')[-3]
    if lab == 'cat' : return [1]
    else : return [0]

def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        lab = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv.resize(cv.imread(path,cv.IMREAD_GRAYSCALE), (IMG_SIZE,IMG_SIZE))
        train_data.append([np.array(img),np.array(lab)])
    shuffle(train_data)
    np.save('train_data.npy',train_data)
    return train_data


# In[2]:


# If you have not run Create_Train_Data():
train_data = create_train_data()

# If you have run Create_Train_Data():
                        #train_data = np.load('train_data.npy',allow_pickle=True)


# In[3]:


train = train_data[:-24500]                      #Segmenting the Data Set
test = train_data[1000:-23750]

train_x = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
train_y = np.array([i[1] for i in train])

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = np.array([i[1] for i in test])

print("Number of Training Samples followed by the Size of each Sample "+ str(train_x.shape))
print("Label of of each Training Sample "+ str(train_y.shape))
print("Number of Testing Samples followed by the Size of each Sample "+ str(test_x.shape))
print("Label of of each Testing Sample "+ str(test_y.shape))


# In[4]:


train_x = train_x.reshape(train_x.shape[0], -1).T        # flattening the images into 1 dimension
test_x = test_x.reshape(test_x.shape[0], -1).T


train_x = train_x / 255         #Normalizing the Values
test_x = test_x / 255


# In[5]:


def relu(Z):                                                       #Helper Functions
    r = np.maximum(0, Z)
    return r,Z

def sigmoid(Z):
    s = 1 / (1 + np.exp(-1 * Z))
    return s,Z

def relu_diff(dA, vals):
    
    Z = vals
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0

    return dZ

def sigmoid_diff(dA, vals):
   
    Z = vals
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ


# In[6]:


def init_param(layer_dims):
    
    parameters = {}                # Initializing the Dictionary
    dim = len(layer_dims)            # Convert the array to int

    for i in range(1, dim):          #loop to access to each dimension
      
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) / 100 #normazling
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
        
    return parameters


# In[7]:


def linear_forward(A, W, b):
    #print("W SHAPE " + str(W.shape),"A SHAPE " + str(A.shape),"b SHAPE " + str(b.shape) )
   
    Z = np.dot(W, A) + b
    vals = (A, W, b)
    
    return Z, vals


# In[8]:


def linear_backward(dZ, vals):
  

    A_prev, W, b = vals
    m = A_prev.shape[1]

    dW = np.dot(dZ, vals[0].T) / m
    db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(vals[1].T, dZ)
    
    return dA_prev, dW, db


# In[9]:


def Back_Pass(AL, Y, vals):
   
    gradients = {}
    L = len(vals) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_val = vals[-1]
    gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = linear_backward(sigmoid_diff(dAL,current_val[1]),
                                                                                       current_val[0])
  
    for l in reversed(range(L-1)):
        current_val = vals[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_diff(dAL, current_val[1]), current_val[0])
        gradients["dA" + str(l + 1)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp
        
    return gradients


# In[10]:


def linear_activation_forward_relu(A_prev, W, b):
    
    Z, linear_vals = linear_forward(A_prev, W, b)
    A, activation_vals = relu(Z)
    
    vals = (linear_vals, activation_vals)          #Used for Back-propopgation 

    return A, vals


# In[11]:


def linear_activation_forward_sig(A_prev, W, b):
    
    Z, linear_vals = linear_forward(A_prev, W, b)
    A, activation_vals = sigmoid(Z)
    
    vals = (linear_vals, activation_vals)          #Used for Back-propopgation 

    return A, vals


# In[12]:


def Frwrd_Pass(X, parameters):
   
    vals = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    #print(parameters)
    for l in range(1, L):                     # Hidden Layers
        A_prev = A 
        A, val = linear_activation_forward_relu(A_prev,parameters['W' + str(l)],parameters['b' + str(l)])
        vals.append(val)
        
    prob, val = linear_activation_forward_sig(A,             # Output Layer Sigmoid for Classification
                                          parameters['W' + str(L)], 
                                          parameters['b' + str(L)])
    
    vals.append(val)
    
    return prob, vals


# In[13]:


def Cross_Entropy_Loss(pred, Y):
    Y = Y.transpose()
    m = Y.shape[1]
    
    loss = (-1 / m) * np.sum(np.multiply(Y, np.log(pred)) + np.multiply(1 - Y, np.log(1 - pred)))     # Cost Func
    
    loss = np.squeeze(loss)      # to convert the matrix into single value
    
    return loss


# In[14]:


def upd_param(parameters, grads, alpha):
   
    # Gradient Descent 
    
    L = len(parameters) // 2  # number of layers in the neural network

    # updating values based on alpha(learning rate)
    
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - alpha * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)]= parameters["b" + str(l + 1)] - alpha * np.array([np.reshape(grads["db" + str(l + 1)],-1)]).T
        
    
    return parameters


# In[15]:


def CNN_4_Binary_Class(X, Y, layers_dims, alpha=0.0075, num_iterations=3000): 
  
    itr_costs = []                                                               # keep track of cost
    
    parameters = init_param(layers_dims)                                     # Initliaze Paremeters          
    
    for i in (trange(0, num_iterations)):                                       # Loop over network

        # Forward Pass
       
        pred, vals = Frwrd_Pass(X, parameters) 
        
        cost = Cross_Entropy_Loss(pred, Y)
        
        # Back Pass
        grads = Back_Pass(pred, Y, vals)
        
        parameters = upd_param(parameters, grads, alpha)
                
        if i % 100 == 0:
            itr_costs.append((i, cost))
            
    
    return parameters,itr_costs


# In[16]:


lay_dims = [4096,11,7,5,1]
parameters, cost = CNN_4_Binary_Class(train_x, train_y, lay_dims, num_iterations=1800)


cost2 = [(elem1, (elem2)) for elem1, elem2 in cost]
plt.scatter(*zip(*cost2))
plt.title('Cost Per Iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

