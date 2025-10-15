# load some packages
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics.pairwise import pairwise_distances, rbf_kernel,  polynomial_kernel, linear_kernel, sigmoid_kernel, laplacian_kernel, chi2_kernel

# Only usable for small sample sizes
def MMD2_biased(X, Y, kernel=rbf_kernel, **kwargs):

    """Compute the biased estimate of Maximum Mean Discrepancy (MMD) between two samples: X and Y.
    
    Parameters:
    - X: np.ndarray of shape (n_samples_X, n_features)
    - Y: np.ndarray of shape (n_samples_Y, n_features)
    - kernel: function, the kernel function to use (default: RBF kernel)
    - **kwargs: additional arguments for the kernel function
    
    Returns:
    - mmd2: float, the biased MMD^2 estimate
    """
    K_XX = kernel(X, X, **kwargs)
    K_YY = kernel(Y, Y, **kwargs)
    K_XY = kernel(X, Y, **kwargs)
    
    mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd2

def rbf_kernel_factory(sigma):
    gamma = 1.0 / (2.0 * sigma**2)
    def k(X, Y):
        # X: (m,d), Y: (n,d)
        X2 = np.sum(X**2, axis=1, keepdims=True)      # (m,1)
        Y2 = np.sum(Y**2, axis=1, keepdims=True).T    # (1,n)
        D2 = X2 + Y2 - 2 * X @ Y.T
        return np.exp(-gamma * D2)
    return k

def median_sigma(X, Y, max_samples=5000, eps=1e-12):
    Z = np.vstack([X, Y])
    if Z.shape[0] > max_samples:
        idx = np.random.choice(Z.shape[0], size=max_samples, replace=False)
        Z = Z[idx]
    D2 = np.square(Z[:,None,:] - Z[None,:,:]).sum(axis=2)
    tri = D2[np.triu_indices_from(D2, k=1)]
    med = np.median(tri[tri > 0])
    return float(np.sqrt(max(med, eps)))

def gram_matrix(X,Y, kernel, ):
    # kernel: callable (X,Y) -> (len(X), len(Y)) Gram Matrix
    return kernel(X,Y)

def mmd2_biased_from_kernel(X,Y, kernel):
    Kxx = gram_matrix(X,X, kernel )
    Kyy = gram_matrix(Y,Y, kernel )
    Kxy = gram_matrix(X,Y, kernel)
    return Kxx.mean() + Kyy.mean() - 2*Kxy.mean()

def random_sampling(Y, n_sample, replace=False):
    Y_sample = np.random.choice(Y.reshape(-1,), size=n_sample, replace=replace)  #with or without replacement?
    return Y_sample
# -------------------------------------------------


def unbiased_Exx(X,sigma):
    """
    Compute E[k(X,X)] exactly without diagonal terms (i.e. unbiased)
    """
    inv2s2= 1.0 /(2.0*sigma**2)
    D2 = np.square(X[:, None,:] - X[None, :, :]).sum(axis=2)
    np.fill_diagonal(D2,np.nan)
    return np.nanmean(np.exp(-inv2s2*D2))

def MC_Eyy(Y, sigma, n_pairs=200_000, rng=None):
    """
    Compute E[k(Y,Y)] by sampling many distinct pairs from Y. (Monte Carlo)
    """
    rng = np.random.default_rng() if rng is None else rng
    n = Y.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n-1, size=n_pairs)
    j = j + (j >= i)  # ensure j != i
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    D2 = np.square(Y[i] - Y[j]).sum(axis=1)
    return np.mean(np.exp(-inv2s2 * D2))

def MC_Exy(X, Y, sigma, n_pairs=1_000_000, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    m, n = X.shape[0], Y.shape[0]
    ix = rng.integers(0, m, size=n_pairs)
    jy = rng.integers(0, n, size=n_pairs)
    diff = X[ix] - Y[jy]
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    d2 = np.einsum('nd,nd->n', diff, diff)
    return float(np.mean(np.exp(-inv2s2 * d2)))

def mean_rbf_xy(X, Y, sigma, block=500_000):
    """
    Compute E[k(X,Y)] exactly without storing full Gram matrix (uses rbf kernel)
    """
    inv2s2 = 1.0 / (2.0 * sigma * sigma)
    m = X.shape[0]; n = Y.shape[0]
    total = 0.0; count = 0
    for s in range(0, n, block):
        Yb = Y[s:s+block]
        X2 = np.sum(X**2, axis=1, keepdims=True)      # (m,1)
        Y2 = np.sum(Yb**2, axis=1, keepdims=True).T   # (1,|b|)
        D2 = X2 + Y2 - 2 * X @ Yb.T
        K = np.exp(-inv2s2 * D2)
        total += K.sum(); count += K.size
    return total / count

#Compute the MMD using the above 4 functions
def mmd2_biased_fast(X, Y, sigma, n_pairs_yy=200_000):
    """
    Suitable for small X and large Y. If X becomes large too using a different galaxy catalogue,
    consider changing this function to compute E_XX also using MC.
    """
    exx = unbiased_Exx(X, sigma) if len(X)<5000 else MC_Eyy(X,sigma,n_pairs_yy)                 # tiny (m~100)
    exy = mean_rbf_xy(X, Y, sigma)               # exact mean, blockwise
    eyy = MC_Eyy(Y, sigma, n_pairs_yy)           # Monte-Carlo
    return exx + eyy - 2.0 * exy

#Helperfunction to find a good value of sigma for the rbf kernel using median heuristic approach
def sigma_median(X, Y, max_y=5000, rng=None, eps=1e-12):
    rng = np.random.default_rng() if rng is None else rng
    Yc = Y if Y.shape[0] <= max_y else rng.choice(Y.ravel(), size=max_y, replace=False).reshape(-1,1)
    Z = np.vstack([X, Yc])
    D2 = (Z[:,None,:] - Z[None,:,:])**2
    tri = D2[np.triu_indices_from(D2[:,:,0], k=1)]
    med = np.median(tri[tri > 0])
    return float(np.sqrt(max(med, eps)))

#Same Helperfunction as above but for very large Y and X is a subsample of Y.
def sigma_median_pairs(Y, n_pairs=1_000_000, rng=None, eps=1e-12):
    """
    Global RBF bandwidth σ by the median heuristic:
      σ^2 ≈ median( ||y_i - y_j||^2 ) over randomly sampled i≠j pairs from Y.
    Works for shape (n,) or (n,d). O(n_pairs·d) time, O(n_pairs) memory.
    """
    rng = np.random.default_rng() if rng is None else rng
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]  # (n,1)

    n = Y.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n - 1, size=n_pairs)
    j = j + (j >= i)  # ensure i != j

    diffs = Y[i] - Y[j]                 # (n_pairs, d)
    d2 = np.einsum('nd,nd->n', diffs, diffs)  # squared norms
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 0.0
    return float(np.sqrt(max(med, eps)))      # returns σ (not σ^2)


def get_MMDs(size_distr, galaxy_pixels, n_gal=100, 
             sigma= None, n_pairs_yy = 200_000,n_pairs_xy=1_000_000, rng=None):
    """
    Input:
    size_distr: Array with the sizes of all galaxies (preferably converted into arcsec). 
                -> catalogue['r50']*pixscale
    galaxy_pix: Array with the number of the pixel at which each galaxy is located.
    n_gal:  Threshold number for which the MMD is computed for a pixel. 
            If there are not enough galaxies in the pixel, the MMD will not be computed.

    Output:
    MMDs: np.array of the MMDs between the galaxy size distribution in a pixel and the whole distr
    pixels: np.array of the pixel numbers for which the MMD was computed. Can later be used to get 
            the corresponding kappa value for these pixels.
    """
    rng = np.random.default_rng() if rng is None else rng
    Y= size_distr.reshape(-1,1) # Sizes of all galaxies in the right format to work with sklearn

    # #Fix sigma globally if not provided
    # if sigma is None:
    #     sigma = sigma_median_pairs(Y,rng=rng)

    #Precompute E[k(Y,Y)] once (using MC)
    Eyy = MC_Eyy(Y,sigma, n_pairs=n_pairs_yy, rng=rng)

    unique_pixels = np.unique(galaxy_pixels)
    mmd_list =[]
    pix_list = []
    # n_pix= hp.nside2npix(nside) # Number of pixels

    for pixel in unique_pixels:
        mask = (galaxy_pixels == pixel)
        m = int(mask.sum())
        if m < max(n_gal, 2):
            continue

        X = size_distr[mask].reshape(-1,1)   # Size of all galaxies in the pixel nr. {pixel}
        
        #E[k(X,X)] - exact for small X, else MC
        if m < 3000:
            Exx = unbiased_Exx(X, sigma)
        else: 
            Exx = MC_Eyy(X,sigma, n_pairs=min(200_000, m*(m-1)), rng=rng)


        # E[k(X,Y)] – exact mean via blocks
        Exy = MC_Exy(X,Y,sigma, n_pairs=n_pairs_xy ,rng=rng)

        mmd2 = Exx + Eyy -2.0*Exy
        mmd = np.sqrt(np.maximum(mmd2, 0.0))
        pix_list.append(pixel)   # List of the pixels for which MMD was computed, later used to get the right kappa values
        mmd_list.append(mmd2)


    return np.array(mmd_list), np.array(pix_list)


# NEW IMPLEMENTATION OF MMD FOR DIFFERENT KERNELS ---------------------------------

def compute_mmd(X, Y, kernel):
    """
    Compute Maximum Mean Discrepancy (MMD) between samples X and Y using a provided kernel.
    
    Parameters:
        X: array-like, shape (n_samples_X, n_features)
        Y: array-like, shape (n_samples_Y, n_features)
        kernel: callable, must support signature kernel(X, Y), returns kernel matrix
        
    Returns:
        mmd: float, MMD value
    """

    X = np.asarray(X)
    Y = np.asarray(Y)
    m = X.shape[0]
    n = Y.shape[0]
    
    K_XX = kernel(X, X)
    K_YY = kernel(Y, Y)
    K_XY = kernel(X, Y)
    
    # Remove diagonal for unbiased estimator
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    
    mmd = (K_XX.sum() / (m * (m - 1))) \
        + (K_YY.sum() / (n * (n - 1))) \
        - (2 * K_XY.sum() / (m * n))
    
    return mmd


def compute_mmd_subsample(X, Y, kernel, size_X=1000, size_Y=1000, n_iter=10, random_state=None):
    """
    Compute MMD between large X and Y by random subsampling.
    Parameters:
        X: array-like (N_X, features), large dataset
        Y: array-like (N_Y, features), large dataset
        kernel: callable kernel (scikit-learn compatible)
        size_X: int, subsample size from X
        size_Y: int, subsample size from Y
        n_iter: int, number of repetitions
        random_state: int or None, reproducibility
    Returns:
        avg_mmd: float, average MMD over n_iter subsamples
        mmd_values: list of individual MMD values
    """
    rng = np.random.default_rng(random_state)
    mmd_values = []
    for i in range(n_iter):
        Xs = rng.choice(X, size_X, replace=False)
        Ys = rng.choice(Y, size_Y, replace=False)
        mmd = compute_mmd(Xs, Ys, kernel)
        mmd_values.append(mmd)
    return np.mean(mmd_values)
