import numpy as np
import pandas
import scipy
import matplotlib.pyplot as plt
from pylab import scatter, show, legend, xlabel, ylabel
from numpy import loadtxt, where

def normalize_features(array):
  
   array_normalized = (array-array.mean())/array.std()
   mu = array.mean()
   sigma = array.std()

   return array_normalized, mu, sigma

#sigmoid function f(x)=1/(1+e^(-x))

def sigmoid(X):


    return 1.0 / (1.0 + np.exp(-X))
   
#cost function
   
def compute_cost(features, values, theta):
    
    m = len(values)
    
    h=sigmoid(features.dot(theta.T))
    
    J = (1.0 / m) * ((-values.T.dot(np.log(h))) - ((1.0 - values.T).dot(np.log(1.0 - h))))
    
  
    return - 1 * J.sum()
   
#gradient

def gradient_descent(features, values, theta, alpha,num_iteration):
    
    m = len(values)
    cost_history = []
    
    
    

    for i in range(num_iteration):
        cost = compute_cost(features, values, theta)
        h = sigmoid(features.dot(theta.T))
        theta = theta - alpha * np.dot((h-values), features)/m
        cost_history.append(cost)
    return theta, pandas.Series(cost_history)

def compute_r_squared(data, predictions):
     
    r_squared = 1 - np.square(data-predictions).sum() / np.square(data-np.mean(data)).sum()

    return r_squared
   
def predict(theta, features, values):
    p = sigmoid(np.dot(features, theta))
    return p>0.5
    
if __name__ == '__main__':
   

    dataframe = pandas.read_csv('diabets.csv')
    #atributes
    features = dataframe[['pregnancy','glucose','arterial pressure','thickness of TC','insulin','body mass index','heredity','age']]
    #goal
    values = dataframe[['diabet']]
    m = len(values)
    print(dataframe)
    features, mu, sigma = normalize_features(features)

    features['ones'] = np.ones(m)
    features_array = np.array(features)
    values_array = np.array(values).flatten()

    #Set values for alpha, number of iterations.
    alpha = 0.01 
    num_iterations = 1000

    #Initialize theta, perform gradient descent
    theta_gradient_descent = np.zeros(len(features.columns))
    theta_gradient_descent, cost_history = gradient_descent(features_array, values_array, theta_gradient_descent,
                                                            alpha, num_iterations)

    predictions=predict(theta_gradient_descent,features,values)
    print(predictions)
    print("R-square = ",compute_r_squared(values_array,predictions))

    '''--------------------------------------'''
    
   
   
   
    

   
