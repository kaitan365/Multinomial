import sys
import numpy as np
from numpy.random import default_rng

from scipy import linalg as la
from scipy.special import softmax
from scipy.stats import chi2, uniform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot

import matplotlib # used in slurm
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from lib_fast_solve import * 

import matplotlib.colors as mcolors
dict = mcolors.TABLEAU_COLORS
colors = list(dict.values())
# my colors
mc1 = colors[1]
mc2 = colors[0]
# %%
import seaborn as sns
sns.set_theme()


#################################
# specifiy the setting
setting =  1
#################################

if setting == 1: 
    n, p, q = 2000, 600, 1
if setting == 2: 
    n, p, q = 3500, 1000, 1
if setting == 3:
    n, p, q = 5000, 1500, 1
if setting == 4:
    n, p, q = 2000, 600, 2
if setting == 5:
    n, p, q = 2000, 600, 4

K = 2

# dist = sys.argv[1]
# omega = sys.argv[2]
# K = int(sys.argv[3])

rep = 1000 # number of repeated experiments
omega = 'true' # true: use Omega[j,j] ; est: use estimated Omega[j,j]
s = int(p/4) # row sparsity of A^*

# distribution of the covariates
dist = 'Gaussian' # Gaussian, Rademacher, Exp1, SNP
print('n',n,'p',p,'K',K,'s',s, 'q',q, dist, omega)

# covariance matrix 
Sigma = la.toeplitz(0.5 ** np.arange(p)) # Toeplitz matrix
L = la.cholesky(Sigma, lower=True)

Omega = la.inv(Sigma)
j = p-1 # fixed null coordinate

# generate coefficient matrix
rng = default_rng(42)
A = np.zeros((p, K))
A[:s, :] = rng.standard_normal(size = (s,K)) # first s rows are nonzero
# ind = rng.choice(p, size=s)
# A[ind, :] = rng.standard_normal(size = (s,K)) # randomly select nonzero rows

A = A @ la.inv(la.sqrtm(A.T @ Sigma @ A)) # ensure A^T Sigma A = I_K
B_star = np.hstack((A, np.zeros((p,1)))) 
B_star[:5, :] # check the last column is zeros

# useful deterministic matrices: P Q R 
M = np.eye(K+1) - np.ones((K+1, K+1))/(K+1)
Q = la.cholesky(M, lower=True)[:,:K]
assert np.allclose(Q @ Q.T, M)
assert np.allclose(Q.T @ Q, np.eye(K))

P = np.vstack((np.eye(K), -np.ones((1, K))))
R = np.vstack((np.eye(K), np.zeros((1, K))))
# %%
def gen_X(seed):
    rng = default_rng(seed) 
    if dist == 'Gaussian':
        X = rng.standard_normal(size=(n, p)) 
    elif dist == 'Rademacher':
        X = 2 * rng.binomial(1, .5, size = (n, p)) - 1
    elif dist == 'Exp1':
        X = rng.exponential(scale=1.0, size=(n, p)) -1
    else: # dist == 'SNP'
        # non-Gaussian covariates: SNPs taking values in {0, 1, 2}, then center and normalize
        X = np.zeros((n,p))
        for j1 in range(p):
            pj = rng.uniform(.25, .75,1)
            plist = np.array([pj**2, 2*pj*(1-pj), (1-pj)**2]).flatten()
            X[:,j1] = rng.choice(3, n, p=plist)
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    
    X = X @ L.T # L L^T = Sigma
    return X

def gen_Y(X, seed):
    rng = default_rng(seed) 
    # generate Y
    # prob_star = softmax(X @ B_star, axis=1)
    # y = [rng.choice(K+1, p=prob) for prob in prob_star] # take values from {0, 1, ..., K}, shape (n,)
    y_2d = np.stack([rng.choice(K+1, size=q, p=softmax(xi @ B_star)) for xi in X])
    yy = y_2d.reshape((n*q,)) # response in repeated measures model, shape (n*q, ) 
    y_one_hot_3d = (np.arange(y_2d.max()+1) == y_2d[...,None]).astype(int)
    assert y_one_hot_3d.shape == (n, q, K+1)
    # y_bar_2d contains the average over 1...q
    y_bar_2d = y_one_hot_3d.mean(axis=1)
    Y = y_bar_2d 
    # Y = np.eye(K+1)[y] # one-hot encoding, shape (n, K+1)
    return yy, Y

# %%
# classical normality result for \sqrt{n}\hat A^T e_j
j = p-1
X = gen_X(42)
prob = softmax(X @ B_star, axis=1) # shape (n, K+1)
E_Hbar = np.mean(
        prob[:, :, np.newaxis] * np.eye(K+1)[np.newaxis, :, :]
        - prob[:,:, np.newaxis] * prob[:, np.newaxis, :], axis=0)
covA = 1/q * Omega[j,j] * la.inv(R.T @ E_Hbar @ R) # using true Omega. 
# %%
# compute the major, minor, angle for drawing ellipsoid
def get_pars(cov):
    # x ~ N(0, cov), calculate minor, major, angle for drawing ellipsoid 
    val, vec = la.eigh(cov)
    c = np.sqrt(chi2.ppf(0.95, 2)) # 2.45 chi^2_2 (0.05) upper quantile of chi^2_2
    minor = c * np.sqrt(val[0]) * vec[:,0]
    major = c * np.sqrt(val[1]) * vec[:,1]
    width = 2 * c * np.sqrt(val[1]) # 2*la.norm(major) 
    height = 2 * c * np.sqrt(val[0])# 2*la.norm(minor)
    # angle between two vectors, range 0 - 180
    a = major
    b = np.array([1,0])
    angle = np.arccos(np.dot(a,b)/(la.norm(a)*la.norm(b))) * 180/np.pi # angle between major and x-axis
    angle = np.minimum(angle, 180-angle)
    if a[0]*a[1]<0:
        angle = 180-angle
    return width, height, angle

def myellipsoid(x, cov1, cov2):
    """
    Create scatter plot of x[:,0] and x[:,1] 
    add a 95%  confdence ellipsoid for random vector from N(0, cov)

    Parameters
    ----------
    x : array_like, shape (n, 2)
    cov: array_like, shape (2, 2)

    Returns
    -------
    matplotlib.patches.Ellipse
    
    """
    
    # true coverage
    a = np.mean(np.diag(x @ la.inv(cov1) @ x.T) <= chi2.ppf(0.95, 2))
    b = np.mean(np.diag(x @ la.inv(cov2) @ x.T) <= chi2.ppf(0.95, 2))

    lw = 2
    width1, height1, angle1 = get_pars(cov1)
    ellipse1 = Ellipse(xy=(0, 0), width=width1, height=height1, 
                        edgecolor='blue', fc='none', lw=lw, angle=angle1, 
                    #    label= '95% confidence ellipse (classical); coverage='+str(np.round(a*100,2))+'%')
                    label= 'classical; coverage:'+str(np.round(a*100,2))+'%')
    axs.add_patch(ellipse1)

    width2, height2, angle2 = get_pars(cov2)
    ellipse2 = Ellipse(xy=(0, 0), width=width2, height=height2, 
                        edgecolor='red', fc='None', lw=lw, angle=angle2, 
                        # label='95% confidence ellipse (modern);  coverage='+str(np.round(b*100,2))+'%')
                        label= 'modern; coverage:'+str(np.round(b*100,2))+'%')
    axs.add_patch(ellipse2)
    axs.legend(loc='upper left', fontsize=18)

# %%
# multinomial logistic regression 
clf = LogisticRegression(penalty='none', fit_intercept=False, multi_class="multinomial")

# %%
def onerun(seed):
    
    # generate X, y, Y with fixed B_star
    # X y, Y = gen_XY(seed=seed, B_star=B_star)
    X = gen_X(seed)
    XX = np.repeat(X, q, axis=0)
    yy, Y = gen_Y(X, seed)

    # X, XX, yy, Y = gen_XY(seed=seed, B_star=B_star)

    # estimate MLE
    clf.fit(XX, yy)
    B_bar = clf.coef_.T # shape (p, 3)
    
    prob_hat = clf.predict_proba(X) # shape (n, K+1)
    G_bar = prob_hat - Y # shape (n, K+1)
    
    # symmetric parametrization
    B_hat = B_bar @ Q # shape (p, K)
    A_hat = B_bar @ P # shape (p, K)
    
#     G = G_bar @ Q
    if (p*(K+1) <= n+p):
        V_bar = compute_V_solve(X=X, pi=prob_hat, checks=False)
    else:
        V_bar = compute_V_Woodberry(X=X, pi=prob_hat, checks=False)
    
#     V = Q.T @ V_bar @ Q
    
    if (omega == 'true'):
        wj = Omega[j,j]
    else: 
        X_j = np.delete(X, j, axis=1)
        a = (np.eye(n) - X_j @ la.inv(X_j.T @ X_j) @ X_j.T) @ X[:,j]
        wj = (n - p + 1) / la.norm(a)**2
        
    # test statistics 
    chi2_old = n * A_hat[j,:] @ la.inv(covA) @ A_hat.T[:,j]
    chi2_new = (wj)**(-1) * A_hat[j,:] @ R.T @ V_bar @ la.pinvh(G_bar.T @ G_bar) @ V_bar @ R @ A_hat.T[:,j]
    
    # a = la.sqrtm(la.pinvh(G_bar.T @ G_bar))
    b = Q @ la.sqrtm(la.inv(Q.T @ G_bar.T @ G_bar @ Q)) @ Q.T
    # assert np.allclose(a, b)
    
    # left multiplicative matrix 
    temp = np.eye(K) + np.ones(K)/(np.sqrt(K+1)+1) # (R^T Q Q^T)^{-1/2}
    Gam_n = (n * wj)**(-1/2) * temp @ R.T @ b @ V_bar @ R
    # the results is Gam_n \sqrt{n} \hat A^T \to N(0, I_K)
    
    return (np.sqrt(n) * A_hat[j,:], Gam_n, [chi2_old, chi2_new])

# %%
a = onerun(1)
a[0], a[1], a[2]
print(a)
# %%
# check the p-value is uniformly distributed
from joblib import Parallel, delayed
output = Parallel(n_jobs=-1, verbose=5)(
    delayed(onerun)(seedid)
    for seedid in range(rep))

dataA = np.zeros((rep, K)) # each row is a realization of $\sqrt{n} \hat B^T e_j$
S = np.zeros((rep, K, K)) # 1000 realizations of $R_B$
test_stats = np.zeros((rep, 2)) # test statistics (old, new)
for i in range(rep):
    dataA[i] = (output[i])[0] 
    S[i] = (output[i])[1]
    test_stats[i] = (output[i])[2]

# %%
S_limit = np.mean(S, axis=0)
covA_new = la.inv(S_limit.T @ S_limit)

# %%
if K==2:
    x, y = np.mgrid[-25:25:0.15, -25:25:0.15]
    pos = np.dstack((x, y))
    average = np.zeros(pos.shape[:-1])
    for i in range(rep):
        random_covariance = S[i].T @ S[i]
        inside = np.einsum('ijx,ijy, xy -> ij', pos, pos, random_covariance) < chi2.ppf(0.951, df=2)
        outside = np.einsum('ijx,ijy, xy -> ij', pos, pos, random_covariance) > chi2.ppf(0.949, df=2)
        average += 1.0 * inside * outside / rep

    # dataA = dataB @ la.inv(T)
    fig, axs = plt.subplots(1,1,figsize=(10,10), dpi=600)
    axs.axis('equal')
    axs.scatter(dataA[:, 0], dataA[:, 1], c ="black", s = 2)
    plt.contour(x, y, average, cmap='Reds', alpha=0.4) # Blues, Oranges
    myellipsoid(dataA, covA, covA_new)
    # plt.xlabel(r'$\sqrt{n} \hat A_{j,1}$', fontsize=15)
    # plt.ylabel(r'$\sqrt{n} \hat A_{j,2}$', fontsize=15)
    ################################
    # auto adjust the axes limits
    lambdas, V = np.linalg.eig(la.inv(covA_new))
    c = chi2.ppf(0.951, df=2)
    a = np.sqrt(c / lambdas[0])
    b = np.sqrt(c / lambdas[1])
    theta = np.arctan2(V[1,0], V[0,0])
    t = np.linspace(0, 2*np.pi, 1000)
    x = a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
    y = a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)
    mg = 3 # margin 
    plt.xlim([np.min(x)-mg, np.max(x)+mg])
    plt.ylim([np.min(y)-mg, np.max(y)+mg])
    # plt.title(r'scatter plot of $\sqrt{n} {\hat A}^T e_j$')
    plt.savefig(f'figs/scatter_{dist}_{omega}_n{n}_p{p}_q{q}.pdf', bbox_inches='tight')
    # plt.show()
# %%