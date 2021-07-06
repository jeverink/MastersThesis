import numpy as np;
import numpy.linalg as linalg;
import numpy.fft as fft;

import pywt;

import general_utils as utils;

def HT(x, sparsity):
    abv = np.absolute(x);
    part = np.argpartition(abv, len(x)-sparsity);
    z = np.copy(x);
    z[part[:len(x)-sparsity]] = 0;
    return z;


def HT_transform(x, sparsity, trans, inv_trans):
    return inv_trans(HT(trans(x), sparsity));



def fourier2d(x):
    """
    Applies 2D fast fourier transform to image. 
    """
    return fft.fft2(x.reshape((28,28))).flatten();

def inverse_fourier2d(x):
    """
    Applies inverse 2D fast fourier transform to image. 
    """
    return fft.ifft2(x.reshape((28,28))).flatten();


def waveletunpack(coef):
    """
    Unpacks tuple of approximation, horizontal detail, vertical detail and
    diagonal detail coefficients into single vector.
    
    """
    
    cA = coef[0].flatten();
    cH = coef[1][0].flatten();
    cV = coef[1][1].flatten();
    cD = coef[1][2].flatten();
    return np.concatenate((cA, cH, cV, cD));

def waveletpack(coef):
    """
    Splits a single vector of wavelet coefficients into separate approximation,
    horizontal detail, vertical detail and diagonal detail coefficients components.
    """
    splits = np.split(coef, 4);
    cA = splits[0].reshape((14,14));
    cH = splits[1].reshape((14,14));
    cV = splits[2].reshape((14,14));
    cD = splits[3].reshape((14,14));
    return (cA, (cH, cV, cD));
    
    
def haar2d(x):
    """
    Applies single level 2D Haar wavelet transform.
    """
    return waveletunpack(pywt.dwt2(x.reshape((28,28)), 'haar'));
    
def inverse_haar2d(x):
    """
    Applies inverse single level 2D Haar wavelet transform.
    """
    return pywt.idwt2(waveletpack(x), 'haar').flatten();


def waveletunpack_cifar(coef):
    cA = coef[0].flatten();
    cH = coef[1][0].flatten();
    cV = coef[1][1].flatten();
    cD = coef[1][2].flatten();
    return np.concatenate((cA, cH, cV, cD));

def waveletpack_cifar(coef):
    splits = np.split(coef, 4);
    cA = splits[0].reshape((16,16));
    cH = splits[1].reshape((16,16));
    cV = splits[2].reshape((16,16));
    cD = splits[3].reshape((16,16));
    return (cA, (cH, cV, cD));
    
def haar2d_3channel(x):
    (c1, c2, c3) = np.split(x, 3);
    u_channel1 = waveletunpack_cifar(pywt.dwt2(c1.reshape((32,32)), 'haar'));
    u_channel2 = waveletunpack_cifar(pywt.dwt2(c2.reshape((32,32)), 'haar'));
    u_channel3 = waveletunpack_cifar(pywt.dwt2(c3.reshape((32,32)), 'haar'));
    return np.concatenate((u_channel1, u_channel2, u_channel3));
    
def inverse_haar2d_3channel(x):
    (c1, c2, c3) = np.split(x, 3);
    channel1 = pywt.idwt2(waveletpack_cifar(c1), 'haar').flatten();
    channel2 = pywt.idwt2(waveletpack_cifar(c2), 'haar').flatten();
    channel3 = pywt.idwt2(waveletpack_cifar(c3), 'haar').flatten();
    return np.concatenate((channel1, channel2, channel3));



def best_normalized_error(x, sparsity):
    """
    Computes normalized distance of a signal to the set of sparse vectors in the standard basis.
    """
    return linalg.norm(utils.normalize(x)-utils.normalize(HT(x, sparsity)));

def best_error_transform(x, sparsity, trans, inv_trans):
    """
    Computes normalized distance of a signal to the set of sparse vectors in any basis defined by
    the transformation and inverse transformation from the standard basis to that basis.
    """
    return linalg.norm(utils.normalize(x)-utils.normalize(HT_transform(x, sparsity, trans, inv_trans)));
    