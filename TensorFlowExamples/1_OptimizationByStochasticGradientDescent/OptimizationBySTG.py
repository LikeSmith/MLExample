"""
OptimizationBySGD.py

Example illustrating Optimization with stochastic gradient descent
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

# Load data
with open('../../data/MSDResponse/MSDResponse_noise_small.csv', 'r') as f:
    T = []
    X = []
    N = 0
    row = f.readline().split(',')
    
    while len(row) == 2:
        T.append(float(row[0]))
        X.append(float(row[1]))
        N += 1
        
        row = f.readline().split(',')
        
    T = np.array(T)
    X = np.array(X)

# Define function to fit
def x(t, p):
    K = p[0]
    tau = p[1]
    w_d = p[2]
    
    return K*(1 - np.exp(-t/tau)*np.cos(t*w_d))

# Define cost
def c_batch(T, X, p):
    c = 0
    for i in range(T.shape[0]):
        c += (X[i] - x(T[i], p))**2
    return c

# Define Gradients
def x_grad(t, p):
    K = p[0]
    tau = p[1]
    w_d = p[2]
    
    x_K =  (1 - np.exp(-t/tau)*np.cos(t*w_d))
    x_tau = -K*t/(tau**2)*np.exp(-t/tau)*np.cos(t*w_d)
    x_w_d = K*w_d*np.exp(-t/tau)*np.sin(t*w_d)
    
    return np.array([x_K, x_tau, x_w_d])

def c_batch_grad(T, X, p):
    grad_c = np.zeros(3)
    for i in range(T.shape[0]):
        grad_c += -2*x_grad(T[i], p)*(X[i] - x(T[i], p))
    return grad_c
    
# training parameters
epochs = 20
batch_size = 10
lr = 0.001
p_0 = [1, 0.1, 10]

if __name__ == '__main__':
    costs = np.zeros(epochs)
    p = np.zeros((epochs+1, len(p_0)))
    p[0, :] = p_0
    
    n_batches = int(np.ceil(N/batch_size))
    
    for epoch in range(epochs):
        print('Epoch: %d/%d'%(epoch+1, epochs))
        perm = np.random.permutation(N)
        
        for batch in trange(n_batches):
            T_batch = T[perm[batch*batch_size:min((batch+1)*batch_size, N)]]
            X_batch = X[perm[batch*batch_size:min((batch+1)*batch_size, N)]]
            costs[epoch] += c_batch(T_batch, X_batch, p[epoch, :])
            grad = c_batch_grad(T_batch, X_batch, p[epoch, :])
            p[epoch, :] -= lr*grad
        
        p[epoch+1, :] = p[epoch, :]
        
    p[1:, :] = p[0:-1, :]
    p[0, :] = p_0

    # evaluate results
    plt.figure()
    plt.plot(T, X, 'b.')
    plt.plot(T, x(T, p[-1, :]), 'r')
    plt.xlabel('time (s)')
    plt.ylabel('displacement')
    plt.legend(('data', 'fitted curve'))
    
    plt.figure()
    plt.plot(costs)
    plt.xlabel('iteration')
    plt.ylabel('cost')
    
    plt.figure()
    plt.plot(p[:, 0])
    plt.plot(p[:, 1])
    plt.plot(p[:, 2])
    plt.xlabel('iteration')
    plt.legend(('K estimate', 'tau estimate', 'omega estimate'))
