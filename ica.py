#%% 
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import FastICA

#%%

def whiten(x):
    x -= np.repeat(np.mean(x, axis=1)[:,None], x.shape[1], axis=1)
    cov = np.cov(x)
    w, v = np.linalg.eig(cov) 
    x = v.T @ x
    return x

def ica(x, niter, eta=0.01):
    '''
    x       - matrix of data, each column is a datapoint
    niter   - number of optimization steps
    '''
    dim, ndata = x.shape
    W = np.eye(dim)
       
    for i in tqdm(range(niter)):
        W_min_T = np.linalg.inv(W.T)
        u = W @ x
        ux = np.zeros((dim,dim))
        for k in range(ndata):
            ux += np.tanh(u[:,k])[None,:] @ x[:,k][:, None]

        W += eta * (W_min_T - (2/ndata) * ux)

    return W

#%%

s = np.array([np.sin(np.linspace(0,10,1000)), np.cos(np.linspace(0,100,1000) + 2000)])
# s = np.random.uniform(size=(2,1000))
A = np.array([[1,2],[2,1]])

x = A @ s
x_white = whiten(x)

plt.figure()
plt.scatter(x[0], x[1], marker='.', color="black", label="data")
plt.scatter(x_white[0], x_white[1], marker='.', color='blue', label="whitened data")
plt.legend()

plt.title("data whitening")
plt.show()

plt.figure()
plt.subplot(211)
plt.title("ground truth signals")
plt.plot(s[0])
plt.plot(s[1])
plt.subplot(212)
plt.title("observed mixtures")
plt.plot(x[0])
plt.plot(x[1])
plt.show()

# %%

W = ica(x_white, 1000, 0.1)
# %%
s_tilde = W @ x_white

plt.figure()
plt.plot(s_tilde[0])
plt.plot(s_tilde[1])
plt.title("source reconstruction")
plt.show()
# %%
icamodel = FastICA(whiten=False)
icamodel.fit(x.T)

# %%
s_tilde_scikit = icamodel.transform(x.T)
plt.figure()
plt.plot(s_tilde_scikit[:,0])
plt.plot(s_tilde_scikit[:,1])
plt.title("source reconstruction scikit")
plt.show()
# %%
