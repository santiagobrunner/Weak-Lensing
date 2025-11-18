# load some packages
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from functools import partial

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
    
    
    mmd2 = (K_XX.sum() / (m * m)) \
        + (K_YY.sum() / (n * n)) \
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

def directed_rbf_kernel_matrix(X, Y, sigma=1.0):
    """Compute directed RBF kernel matrix."""
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    delta = X - Y.T
    return np.sign(delta) * np.exp(-(delta ** 2) / (2 * sigma ** 2))

def sigma_median_pairs(Y, n_pairs=1_000_000, rng=None, eps=1e-12):
    rng = np.random.default_rng(rng)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = Y.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n-1, size=n_pairs); j += (j >= i)
    d = Y[i] - Y[j]                       # (n_pairs, d)
    d2 = np.einsum('nd,nd->n', d, d)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else eps
    return float(np.sqrt(max(med, eps)))

# Load the catalogue and power spectrum
catalogue1000 = np.load('catalogue_1000sqd.npy')
cl_kappa_225 = np.loadtxt('cl_kappa_mean_225.txt')[:,1] # load power spectrum
cl_kappa_225 = np.concatenate((np.zeros(2), cl_kappa_225)) # add zeros for monopole and dipole

# Create Coarser Kappa Map
nside = 64 # HEALPix nside parameter
coarse_random_seed = 42
np.random.seed(coarse_random_seed)
kappamap_225 = hp.synfast(cl_kappa_225, nside)  # generate kappa map from power spectrum
print("Cl_kappa225 shape:", cl_kappa_225.shape, "   kappamap225 shape:", kappamap_225.shape )

pixscale = 0.263
sizes_in_arcsec1000 = catalogue1000['r50'] * pixscale   #arcsec

# Convert galaxy coordinates to HEALPix pixel indices
galaxy_pix1000 = hp.ang2pix(nside, catalogue1000['ra'], catalogue1000['dec'], lonlat=True)
galaxy_pix1000_unique, galaxy_pix1000_counts = np.unique(galaxy_pix1000, return_counts=True)
n_pixels = hp.nside2npix(nside)

intrinsic_size1000 = sizes_in_arcsec1000
observed_size1000 = sizes_in_arcsec1000 * (1.0 + kappamap_225[galaxy_pix1000])

size_mask1000 = (intrinsic_size1000 < 5.0) #arcsec

# Create a finer kappa map for recovery
nside_fine = 1024
recovery_rand_seed = 31
np.random.seed(recovery_rand_seed)
recovery_kappamap = hp.synfast(cl_kappa_225, nside_fine)

# Convert galaxy coordinates to pixel numbers in the finer map
gal_pix_fine = hp.ang2pix(nside_fine, catalogue1000['ra'], catalogue1000['dec'], lonlat=True)
gal_pix_fine_unique, gal_pix_fine_counts = np.unique(gal_pix_fine, return_counts=True)

recovery_observed_size = intrinsic_size1000 * (1.0 + recovery_kappamap[gal_pix_fine])


# Compute the MMDs for each bigger/coarser pixel but the new observed sizes.
# Compute also the averaged kappa values for each bigger pixel.


sigma = sigma_median_pairs(recovery_observed_size[size_mask1000], n_pairs=1_000_000, rng=42)
directed_rbf_kernel = partial(directed_rbf_kernel_matrix, sigma=sigma)

kappa_recov_avg_fine = []


batch_size = 125
n_pixels = len(galaxy_pix1000_unique)
Y_lensed1000 = recovery_observed_size[size_mask1000].reshape(-1, 1)

for i, batch_start in enumerate(range(0, n_pixels, batch_size)):    #Iterate over batches
    batch_end = min(batch_start + batch_size, n_pixels)
    pixel_batch = galaxy_pix1000_unique[batch_start:batch_end]
    mmd2_lensed_batch_dir_rbf = []
    mmd2_lensed_batch_rbf = []
    print(f"Starting with batch {i+1}.")
    for p in pixel_batch:  #Iterate over bigger pixels in the batch
        mask = (galaxy_pix1000 == p) 
        index = np.where(galaxy_pix1000_unique == p)[0][0]
        print(f"Pixel {index} / {n_pixels}")

        kappa_values = recovery_kappamap[gal_pix_fine[mask]]    # Compute the mean kappa value for the bigger pixel
        kappa_avg = np.mean(kappa_values)
        kappa_recov_avg_fine.append(kappa_avg)

        if mask.sum() > 20000:
            X_lensed_fine = recovery_observed_size[mask & size_mask1000].reshape(-1, 1)
            mmd2_dir = compute_mmd_subsample(X_lensed_fine, Y_lensed1000, directed_rbf_kernel, 20000,20000,3,42)
            mmd2_rbf = compute_mmd_subsample(X_lensed_fine, Y_lensed1000, rbf_kernel, 20000,20000,3,42)
            mmd2_lensed_batch_dir_rbf.append(mmd2_dir)
            mmd2_lensed_batch_rbf.append(mmd2_rbf)

    # Save batch results
    np.save(f'/cluster/home/sbrunne/mmd2_recov_dir_rbf/mmd2_lensed_dir_rbf_fine_batch_{i+1}.npy', mmd2_lensed_batch_dir_rbf)
    np.save(f'/cluster/home/sbrunne/mmd2_recov_rbf/mmd2_lensed_rbf_fine_batch_{i+1}.npy', mmd2_lensed_batch_rbf)
    print(f"Batch {i+1} saved!")