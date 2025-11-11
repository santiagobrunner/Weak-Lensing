# load some packages
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel

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
    # np.fill_diagonal(K_XX, 0)
    # np.fill_diagonal(K_YY, 0)
    
    mmd2 = (K_XX.sum() / (m * (m - 1))) \
        + (K_YY.sum() / (n * (n - 1))) \
        - (2 * K_XY.sum() / (m * n))
    
    return mmd2

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

# load the catalogue
catalogue1000 = np.load('catalogue_1000sqd.npy')

nside = 64 # HEALPix nside parameter

cl_kappa_225 = np.loadtxt('cl_kappa_mean_225.txt')[:,1] # load power spectrum
cl_kappa_225 = np.concatenate((np.zeros(2), cl_kappa_225)) # add zeros for monopole and dipole
kappamap_225 = hp.synfast(cl_kappa_225, nside)  # generate kappa map from power spectrum
print("Cl_kappa225 shape:", cl_kappa_225.shape, "   kappamap225 shape:", kappamap_225.shape )

pixscale = 0.263
sizes_in_arcsec1000 = catalogue1000['r50'] * pixscale   #arcsec

# Convert galaxy coordinates to HEALPix pixel indices
galaxy_pix1000= hp.ang2pix(nside, catalogue1000['ra'], catalogue1000['dec'], lonlat=True)
galaxy_pix1000_unique, galaxy_pix1000_counts = np.unique(galaxy_pix1000, return_counts=True)
n_pixels = hp.nside2npix(nside)

intrinsic_size1000 = sizes_in_arcsec1000
observed_size1000 = sizes_in_arcsec1000 * (1.0 + kappamap_225[galaxy_pix1000])
size_mask1000 = (intrinsic_size1000 < 5.0) #arcsec

# Create a smoother convergence map.
# From now on, we only use the big catalogue
nside_fine = 1024
kappamap_fine = hp.synfast( cl_kappa_225, nside_fine)

# Convert galaxy coordinates to pixel numbers in the finer map
gal_pix_fine = hp.ang2pix(nside_fine, catalogue1000['ra'], catalogue1000['dec'], lonlat=True)
gal_pix_fine_unique, gal_pix_fine_counts = np.unique(gal_pix_fine, return_counts=True)

# Compute observed sizes for all galaxies using the finer kappa map
observed_size_fine = intrinsic_size1000 * (1.0 + kappamap_fine[gal_pix_fine])

# Compute Noise Level for RBF kernel MMD via Subsampling for the new convergence map
mmd2_rbf_noise = []

rng = np.random.default_rng(42)

size = 20_000  # match your per-pixel (or subsample) size
for _ in range(1000):
    Xs = rng.choice(observed_size_fine[size_mask1000], size=size, replace=False)
    Ys = rng.choice(observed_size_fine[size_mask1000], size=size, replace=False)
    mmd2_rbf_noise.append(compute_mmd_subsample(Xs.reshape(-1,1), Ys.reshape(-1,1), rbf_kernel, size, size,3,42))

mmd2_rbf_noise = np.array(mmd2_rbf_noise)
np.save('/cluster/home/sbrunne/mmd2_rbf_noise_fine.npy', mmd2_rbf_noise)
print("mmd2_rbf_noise_fine saved!")